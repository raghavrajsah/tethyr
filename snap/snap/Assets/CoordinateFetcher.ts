@component
export class ARStreamingClient extends BaseScriptComponent {
    // Modules
    private cameraModule: CameraModule = require('LensStudio:CameraModule');
    private internetModule: InternetModule = require('LensStudio:InternetModule');
    
    // Camera related
    private cameraTexture: Texture;
    private cameraTextureProvider: CameraTextureProvider;
    
    // Audio related
    @input
    @hint('Audio From Microphone asset')
    microphoneAudio: AudioTrackAsset | undefined;
    
    private microphoneProvider: MicrophoneAudioProvider | null = null;
    private audioSampleRate: number = 16000;
    
    // WebSocket
    private socket: WebSocket | null = null;
    private isConnected: boolean = false;
    
    @input
    @hint('WebSocket server URL, e.g., ws://192.168.1.100:5000')
    serverUrl: string = 'ws://192.168.1.100:5000';
    
    // UI Elements
    @input
    @hint('Text overlay for displaying instructions')
    overlayText: Text | undefined;
    
    @input
    @hint('Image component for bounding box visualization')
    bboxImage: Image | undefined;
    
    @input
    @hint('Screen transform for positioning bbox')
    screenTransform: ScreenTransform | undefined;
    
    // Frame processing control
    private isProcessingFrame: boolean = false;
    private frameCounter: number = 0;
    
    @input
    @hint('Send every Nth video frame')
    videoFrameSkip: number = 15;
    
    @input
    @hint('Send audio every Nth frame')
    audioFrameSkip: number = 10;

    onAwake() {
        this.createEvent('OnStartEvent').bind(() => {
            this.initialize();
        });
        
        this.createEvent('UpdateEvent').bind(() => {
            this.onUpdate();
        });
        
        this.createEvent('OnDestroyEvent').bind(() => {
            this.cleanup();
        });
    }
    
    private initialize(): void {
        this.initializeCamera();
        this.initializeAudio();
        this.connectWebSocket();
    }
    
    private initializeCamera(): void {
        const cameraRequest = CameraModule.createCameraRequest();
        cameraRequest.cameraId = CameraModule.CameraId.Default_Color;
        
        this.cameraTexture = this.cameraModule.requestCamera(cameraRequest);
        this.cameraTextureProvider = this.cameraTexture.control as CameraTextureProvider;
        
        this.cameraTextureProvider.onNewFrame.add((cameraFrame) => {
            this.onNewCameraFrame(cameraFrame);
        });
        
        print('Camera initialized');
    }
    
    private initializeAudio(): void {
        if (!this.microphoneAudio) {
            print('Warning: No microphone audio asset assigned');
            return;
        }
        
        const control = this.microphoneAudio.control;
        
        if (control.isOfType('Provider.MicrophoneAudioProvider')) {
            this.microphoneProvider = control as MicrophoneAudioProvider;
            this.microphoneProvider.sampleRate = this.audioSampleRate;
            this.microphoneProvider.start();
            print(`Microphone initialized and started at ${this.audioSampleRate}Hz`);
            print(`Max frame size: ${this.microphoneProvider.maxFrameSize}`);
        } else {
            print('Error: Audio asset is not a microphone provider');
        }
    }
    
    private connectWebSocket(): void {
        print(`Connecting to WebSocket: ${this.serverUrl}`);
        
        this.socket = this.internetModule.createWebSocket(this.serverUrl);
        this.socket.binaryType = 'blob';
        
        this.socket.onopen = (event: WebSocketEvent) => {
            print('WebSocket connected');
            this.isConnected = true;
            
            // Send initial handshake
            this.sendMessage({
                type: 'handshake',
                timestamp: Date.now(),
                device: 'spectacles',
                capabilities: {
                    video: true,
                    audio: this.microphoneProvider !== null
                }
            });
        };
        
        this.socket.onmessage = async (event: WebSocketMessageEvent) => {
            await this.handleServerMessage(event);
        };
        
        this.socket.onclose = (event: WebSocketCloseEvent) => {
            this.isConnected = false;
            if (event.wasClean) {
                print('WebSocket closed cleanly');
            } else {
                print(`WebSocket closed with error, code: ${event.code}`);
            }
        };
        
        this.socket.onerror = (event: WebSocketEvent) => {
            print('WebSocket error');
            this.isConnected = false;
        };
    }
    
    private onNewCameraFrame(cameraFrame: CameraFrame): void {
        if (!this.isConnected || this.isProcessingFrame) {
            return;
        }
        
        this.frameCounter++;
        
        // Send video frames
        if (this.frameCounter % this.videoFrameSkip === 0) {
            this.sendVideoFrame();
        }
    }
    
    private onUpdate(): void {
        if (!this.isConnected) {
            return;
        }
        
        // Send audio chunks periodically
        if (this.frameCounter % this.audioFrameSkip === 0) {
            this.sendAudioChunk();
        }
    }
    
    private sendVideoFrame(): void {
        this.isProcessingFrame = true;
        
        Base64.encodeTextureAsync(
            this.cameraTexture,
            (encodedString: string) => {
                const camera = global.deviceInfoSystem.getTrackingCameraForId(
                    CameraModule.CameraId.Default_Color
                );
                const resolution = camera.resolution;
                
                this.sendMessage({
                    type: 'video_frame',
                    data: encodedString,
                    timestamp: Date.now(),
                    frame_number: this.frameCounter,
                    resolution: {
                        width: resolution.x,
                        height: resolution.y
                    }
                });
                
                this.isProcessingFrame = false;
            },
            () => {
                print('Failed to encode video frame');
                this.isProcessingFrame = false;
            },
            CompressionQuality.IntermediateQuality,
            EncodingType.Jpg
        );
    }
    
    // FIXME: The audio chunk is ENTIRELY not sending
    private sendAudioChunk(): void {
        if (!this.microphoneProvider) {
            return;
        }
        
        try {
            const maxSamples = this.microphoneProvider.maxFrameSize;
            
            if (maxSamples <= 0) {
                print('Warning: maxFrameSize is 0 or negative');
                return;
            }
            
            const audioData = new Float32Array(maxSamples);
            const audioShape = this.microphoneProvider.getAudioFrame(audioData);

            const samples = audioShape.x;
            const channels = audioShape.y;
            const depth = audioShape.z;
            
            // Debug output (remove after fixing)
            if (this.frameCounter % 60 === 0) {
                print(`Audio shape: samples=${samples}, channels=${channels}, depth=${depth}, maxFrameSize=${maxSamples}`);
            }
            
            if (samples <= 0) {
                // No audio data available yet
                return;
            }
            
            // Only send the actual samples that were captured, not the entire buffer
            const actualAudioData = audioData.slice(0, samples * channels);
            
            // Encode audio data as base64 for transmission
            const audioBytes = new Uint8Array(actualAudioData.length * 4);
            for (let i = 0; i < actualAudioData.length; i++) {
                const view = new DataView(audioBytes.buffer, i * 4, 4);
                view.setFloat32(0, actualAudioData[i], true); // little-endian
            }
            
            const base64Audio = Base64.encode(audioBytes);
            
            this.sendMessage({
                type: 'audio_chunk',
                data: base64Audio,
                timestamp: Date.now(),
                frame_number: this.frameCounter,
                sample_rate: this.audioSampleRate,
                samples: samples,
                channels: channels
            });
            
            if (this.frameCounter % 60 === 0) {
                print(`Sent audio chunk: ${samples} samples, ${channels} channels`);
            }
        } catch (error) {
            print(`Error capturing audio: ${error}`);
        }
    }
    
    private sendMessage(data: any): void {
        if (this.socket && this.isConnected) {
            try {
                this.socket.send(JSON.stringify(data));
            } catch (error) {
                print(`Error sending message: ${error}`);
            }
        }
    }
    
    private async handleServerMessage(event: WebSocketMessageEvent): Promise<void> {
        try {
            let messageText: string;
            
            if (event.data instanceof Blob) {
                messageText = await event.data.text();
            } else {
                messageText = event.data;
            }
            
            const message = JSON.parse(messageText);
            
            switch (message.type) {
                case 'overlay':
                    this.handleOverlayInstruction(message);
                    break;
                    
                case 'bbox':
                    this.handleBboxInstruction(message);
                    break;
                    
                case 'clear':
                    this.clearAllOverlays();
                    break;
                    
                case 'handshake_ack':
                    print('Server acknowledged connection');
                    break;
                    
                default:
                    print(`Unknown message type: ${message.type}`);
            }
        } catch (error) {
            print(`Error handling server message: ${error}`);
        }
    }
    
    private handleOverlayInstruction(message: any): void {
        if (this.overlayText && message.text) {
            this.overlayText.text = message.text;
            print(`Overlay updated: ${message.text}`);
            
            // Optional: apply styling if provided
            if (message.color) {
                const color = message.color;
                this.overlayText.textFill.color = new vec4(
                    color.r, color.g, color.b, color.a
                );
            }
            
            // Optional: apply positioning if provided
            if (message.position && this.overlayText.getSceneObject()) {
                const screenTransform = this.overlayText.getSceneObject()
                    .getComponent('Component.ScreenTransform') as ScreenTransform;
                if (screenTransform) {
                    screenTransform.anchors.setCenter(
                        new vec2(message.position.x, message.position.y)
                    );
                }
            }
        }
    }
    
    private handleBboxInstruction(message: any): void {
        if (!this.bboxImage || !this.screenTransform || !message.bbox) {
            return;
        }
        
        const bbox = message.bbox;
        const camera = global.deviceInfoSystem.getTrackingCameraForId(
            CameraModule.CameraId.Default_Color
        );
        const resolution = camera.resolution;
        
        // Convert pixel coordinates to normalized screen coordinates (0-1)
        const left = bbox.x / resolution.x;
        const top = bbox.y / resolution.y;
        const width = bbox.width / resolution.x;
        const height = bbox.height / resolution.y;
        
        // Position and scale the bbox image
        this.screenTransform.anchors.setCenter(new vec2(left + width / 2, top + height / 2));
        this.screenTransform.anchors.setSize(new vec2(width, height));
        
        // Make visible
        this.bboxImage.enabled = true;
        
        print(`BBox positioned: ${bbox.label || 'Object'} at (${bbox.x}, ${bbox.y})`);
        
        // Optional: apply color if provided
        if (message.color && this.bboxImage.mainMaterial) {
            const color = message.color;
            this.bboxImage.mainMaterial.mainPass.baseColor = new vec4(
                color.r, color.g, color.b, color.a
            );
        }
    }
    
    private clearAllOverlays(): void {
        if (this.overlayText) {
            this.overlayText.text = '';
        }
        if (this.bboxImage) {
            this.bboxImage.enabled = false;
        }
        print('Overlays cleared');
    }
    
    private cleanup(): void {
        if (this.microphoneProvider) {
            this.microphoneProvider.stop();
        }
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
    }
}