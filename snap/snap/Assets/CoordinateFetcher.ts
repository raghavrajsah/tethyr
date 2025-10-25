/**
 * ARStreamingClient
 *
 * Streams video and audio from Spectacles to a WebSocket server for processing.
 *
 * Audio Configuration:
 * - For MICROPHONE input: Assign only the microphoneAudio input
 * - For PRE-RECORDED audio: Assign both prerecordedAudio AND audioComponent inputs
 *   - The audioComponent must reference the same audio track for proper playback synchronization
 *   - Pre-recorded audio will loop indefinitely (-1 loop count)
 *
 * Timing:
 * - Video frames are sent every Nth frame (configurable via videoFrameSkip)
 * - Audio is read at a fixed time interval (configurable via audioReadIntervalMs)
 * - Default audio interval is 100ms (10 reads/sec) to prevent speed-up issues
 */
@component
export class ARStreamingClient extends BaseScriptComponent {
    // Modules
    private cameraModule: CameraModule = require("LensStudio:CameraModule");
    private internetModule: InternetModule = require("LensStudio:InternetModule");

    // Camera related
    private cameraTexture: Texture;
    private cameraTextureProvider: CameraTextureProvider;

    // Audio related
    @input
    @hint(
        "Audio From Microphone asset (leave empty if using prerecorded audio)"
    )
    microphoneAudio: AudioTrackAsset | undefined;

    @input
    @hint(
        "Pre-recorded Audio Track asset for testing (leave empty if using microphone)"
    )
    prerecordedAudio: AudioTrackAsset | undefined;

    @input
    @hint(
        "Audio Component for playing pre-recorded audio (required if using prerecorded audio)"
    )
    audioComponent: AudioComponent | undefined;

    private microphoneProvider: MicrophoneAudioProvider | null = null;
    private fileAudioProvider: FileAudioTrackProvider | null = null;
    private audioSampleRate: number = 16000;
    private audioReadBuffer: Float32Array | null = null;
    private usePrerecordedAudio: boolean = false;
    private lastAudioReadTime: number = 0;

    // WebSocket
    private socket: WebSocket | null = null;
    private isConnected: boolean = false;

    @input
    @hint("WebSocket server URL, e.g., ws://192.168.1.100:5001")
    serverUrl: string = "ws://172.20.0.231:5001";

    // UI Elements
    @input
    @hint("Text overlay for displaying instructions")
    overlayText: Text | undefined;

    @input
    @hint("BoundBox prefab for bounding box visualization")
    boundBoxPrefab: ObjectPrefab;

    @input
    @hint("Parent scene object for instantiated bounding boxes")
    bboxParent: SceneObject;

    // Active bounding boxes tracking
    private activeBoundBoxes: Map<string, SceneObject> = new Map();

    // Frame processing control
    private isProcessingFrame: boolean = false;
    private frameCounter: number = 0;

    @input
    @hint("Send every Nth video frame")
    videoFrameSkip: number = 15;

    @input
    @hint("Audio read interval in milliseconds (default 100ms = 10 reads per second)")
    audioReadIntervalMs: number = 100;

    onAwake() {
        this.createEvent("OnStartEvent").bind(() => {
            this.initialize();
        });

        this.createEvent("UpdateEvent").bind(() => {
            this.onUpdate();
        });

        this.createEvent("OnDestroyEvent").bind(() => {
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
        this.cameraTextureProvider = this.cameraTexture
            .control as CameraTextureProvider;

        this.cameraTextureProvider.onNewFrame.add((cameraFrame) => {
            this.onNewCameraFrame(cameraFrame);
        });

        print("Camera initialized");
    }

    private initializeAudio(): void {
        // Check if using pre-recorded audio for testing
        if (this.prerecordedAudio && this.audioComponent) {
            this.usePrerecordedAudio = true;
            const control = this.prerecordedAudio.control;

            if (control.isOfType("Provider.FileAudioTrackProvider")) {
                this.fileAudioProvider = control as FileAudioTrackProvider;
                this.fileAudioProvider.sampleRate = this.audioSampleRate;

                print(
                    `Pre-recorded audio initialized at ${this.audioSampleRate}Hz`
                );
                print(`Max frame size: ${this.fileAudioProvider.maxFrameSize}`);

                this.audioReadBuffer = new Float32Array(
                    this.fileAudioProvider.maxFrameSize
                );

                // Set the audio track on the component
                this.audioComponent.audioTrack = this.prerecordedAudio;

                // Start playing the audio on loop for continuous testing
                // -1 means loop indefinitely
                this.audioComponent.play(-1);

                print("Pre-recorded audio started playing (looping indefinitely)");
                print(`Audio component is playing: ${this.audioComponent.isPlaying()}`);

                // Calculate expected samples per read interval
                const samplesPerInterval = (this.audioSampleRate * this.audioReadIntervalMs) / 1000;
                print(`Expected samples per ${this.audioReadIntervalMs}ms: ~${samplesPerInterval}`);
            } else {
                print(
                    "Error: Prerecorded audio asset is not a file audio provider"
                );
                this.usePrerecordedAudio = false;
            }
            return;
        }

        // Otherwise, use microphone
        if (!this.microphoneAudio) {
            print(
                "Warning: No audio source assigned (neither microphone nor prerecorded)"
            );
            return;
        }

        const control = this.microphoneAudio.control;

        if (control.isOfType("Provider.MicrophoneAudioProvider")) {
            this.microphoneProvider = control as MicrophoneAudioProvider;
            this.microphoneProvider.sampleRate = this.audioSampleRate;
            this.microphoneProvider.start();
            print(
                `Microphone initialized and started at ${this.audioSampleRate}Hz`
            );
            print(`Max frame size: ${this.microphoneProvider.maxFrameSize}`);
            this.audioReadBuffer = new Float32Array(
                this.microphoneProvider.maxFrameSize
            );
        } else {
            print("Error: Audio asset is not a microphone provider");
        }
    }

    private connectWebSocket(): void {
        print(`Connecting to WebSocket: ${this.serverUrl}`);

        this.socket = this.internetModule.createWebSocket(this.serverUrl);
        this.socket.binaryType = "blob";

        this.socket.onopen = (event: WebSocketEvent) => {
            print("WebSocket connected");
            this.isConnected = true;

            // Send initial handshake
            this.sendMessage({
                type: "handshake",
                timestamp: Date.now(),
                device: "spectacles",
                capabilities: {
                    video: true,
                    audio:
                        this.microphoneProvider !== null ||
                        this.fileAudioProvider !== null,
                    audioSource: this.usePrerecordedAudio
                        ? "prerecorded"
                        : "microphone",
                },
            });
        };

        this.socket.onmessage = async (event: WebSocketMessageEvent) => {
            await this.handleServerMessage(event);
        };

        this.socket.onclose = (event: WebSocketCloseEvent) => {
            this.isConnected = false;
            if (event.wasClean) {
                print("WebSocket closed cleanly");
            } else {
                print(
                    `WebSocket closed with error, code: ${event.code} ${event.reason}`
                );
            }
        };

        this.socket.onerror = (event: WebSocketEvent) => {
            print("WebSocket error");
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

        // Time-based throttling for audio reading to prevent speed-up
        const currentTime = Date.now();
        const timeSinceLastRead = currentTime - this.lastAudioReadTime;

        // Only read audio at the configured interval (default 100ms = 10 times per second)
        if (timeSinceLastRead < this.audioReadIntervalMs) {
            return;
        }

        // For pre-recorded audio, check if the audio component is actually playing
        if (this.usePrerecordedAudio) {
            if (!this.audioComponent || !this.audioComponent.isPlaying()) {
                // Audio is not playing, don't read frames
                return;
            }
        }

        // Read audio from the appropriate source
        let audioProvider:
            | MicrophoneAudioProvider
            | FileAudioTrackProvider
            | null = null;

        if (this.usePrerecordedAudio && this.fileAudioProvider) {
            audioProvider = this.fileAudioProvider;
        } else if (this.microphoneProvider) {
            audioProvider = this.microphoneProvider;
        }

        if (!audioProvider || !this.audioReadBuffer) {
            return;
        }

        const audioShape = audioProvider.getAudioFrame(this.audioReadBuffer);
        const samples = audioShape.x;
        const channels = audioShape.y;

        if (samples > 0) {
            this.sendAudioChunk(this.audioReadBuffer, samples, channels);
            this.lastAudioReadTime = currentTime;
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
                    type: "video_frame",
                    data: encodedString,
                    timestamp: Date.now(),
                    frame_number: this.frameCounter,
                    resolution: {
                        width: resolution.x,
                        height: resolution.y,
                    },
                });

                this.isProcessingFrame = false;
            },
            () => {
                print("Failed to encode video frame");
                this.isProcessingFrame = false;
            },
            CompressionQuality.IntermediateQuality,
            EncodingType.Jpg
        );
    }

    private sendAudioChunk(
        audioData: Float32Array,
        samples: number,
        channels: number
    ): void {
        if (!this.microphoneProvider && !this.fileAudioProvider) {
            // Extra guard - no audio source available
            return;
        }

        try {
            // Debug output - log every 10 audio chunks
            if (this.frameCounter % 10 === 0) {
                const source = this.usePrerecordedAudio
                    ? "prerecorded"
                    : "microphone";
                const durationMs = (samples / this.audioSampleRate) * 1000;
                print(
                    `Sending audio (${source}): samples=${samples}, channels=${channels}, duration=${durationMs.toFixed(1)}ms`
                );
            }

            // Only send the actual samples that were captured, not the entire buffer
            // (audioData is the *full* buffer, so we slice from 0 to what was filled)
            const actualAudioData = audioData.slice(0, samples * channels);

            // Encode audio data as base64 for transmission
            const audioBytes = new Uint8Array(actualAudioData.length * 4);
            // Create the DataView *once* outside the loop for efficiency
            const view = new DataView(audioBytes.buffer);
            for (let i = 0; i < actualAudioData.length; i++) {
                view.setFloat32(i * 4, actualAudioData[i], true); // little-endian
            }

            const base64Audio = Base64.encode(audioBytes);

            this.sendMessage({
                type: "audio_chunk",
                data: base64Audio,
                timestamp: Date.now(),
                frame_number: this.frameCounter,
                sample_rate: this.audioSampleRate,
                samples: samples,
                channels: channels,
            });
        } catch (error) {
            print(`Error sending audio: ${error}`);
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

    private async handleServerMessage(
        event: WebSocketMessageEvent
    ): Promise<void> {
        try {
            let messageText: string;

            if (event.data instanceof Blob) {
                messageText = await event.data.text();
            } else {
                messageText = event.data;
            }

            const message = JSON.parse(messageText);

            switch (message.type) {
                case "overlay":
                    this.handleOverlayInstruction(message);
                    break;

                case "bbox":
                    this.handleBboxInstruction(message);
                    break;

                case "clear":
                    this.clearAllOverlays();
                    break;

                case "handshake_ack":
                    print("Server acknowledged connection");
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
                    color.r,
                    color.g,
                    color.b,
                    color.a
                );
            }

            // Optional: apply positioning if provided
            if (message.position && this.overlayText.getSceneObject()) {
                const screenTransform = this.overlayText
                    .getSceneObject()
                    .getComponent(
                        "Component.ScreenTransform"
                    ) as ScreenTransform;
                if (screenTransform) {
                    screenTransform.anchors.setCenter(
                        new vec2(message.position.x, message.position.y)
                    );
                }
            }
        }
    }

    private handleBboxInstruction(message: any): void {
        if (!message.bbox) {
            return;
        }

        const bbox = message.bbox;
        const bboxKey = `${bbox.label}_${bbox.x}_${bbox.y}`;
        this.createOrUpdateBoundBox(bboxKey, bbox, message.color);
    }

    private createOrUpdateBoundBox(
        bboxKey: string,
        bbox: any,
        color: any
    ): void {
        const camera = global.deviceInfoSystem.getTrackingCameraForId(
            CameraModule.CameraId.Default_Color
        );
        const resolution = camera.resolution;

        // Check if we already have this bounding box
        let bboxInstance = this.activeBoundBoxes.get(bboxKey);

        // Create new instance if needed
        if (!bboxInstance) {
            bboxInstance = this.boundBoxPrefab.instantiate(this.bboxParent);
            this.activeBoundBoxes.set(bboxKey, bboxInstance);
            print(`Created new BoundBox for: ${bbox.label}`);
        }

        // Get the ScreenTransform component from the main object
        const screenTransform = bboxInstance.getComponent(
            "Component.ScreenTransform"
        ) as ScreenTransform;

        if (screenTransform) {
            // Convert pixel coordinates to normalized screen coordinates (0-1)
            const left = bbox.x / resolution.x;
            const top = bbox.y / resolution.y;
            const width = bbox.width / resolution.x;
            const height = bbox.height / resolution.y;

            // Set the anchors to position the bbox
            // In Lens Studio, anchors define the normalized position
            screenTransform.anchors.left = left;
            screenTransform.anchors.right = left + width;
            screenTransform.anchors.top = top;
            screenTransform.anchors.bottom = top + height;

            print(
                `BBox positioned: ${bbox.label} at (${bbox.x}, ${bbox.y}) size (${bbox.width}x${bbox.height})`
            );
        }

        // Get the Image component and set color
        const imageComponent = bboxInstance.getComponent(
            "Component.Image"
        ) as Image;

        if (imageComponent && color && imageComponent.mainMaterial) {
            imageComponent.mainMaterial.mainPass.baseColor = new vec4(
                color.r,
                color.g,
                color.b,
                color.a
            );
        }

        // Find the Label child object and set the text
        const childCount = bboxInstance.getChildrenCount();
        for (let i = 0; i < childCount; i++) {
            const child = bboxInstance.getChild(i);
            if (child.name === "Label") {
                const textComponent = child.getComponent(
                    "Component.Text"
                ) as Text;
                if (textComponent) {
                    textComponent.text = bbox.label || "Object";
                    print(`Set label text to: ${bbox.label}`);
                }
                break;
            }
        }

        // Make sure the instance is enabled
        bboxInstance.enabled = true;
    }

    private clearAllOverlays(): void {
        if (this.overlayText) {
            this.overlayText.text = "";
        }
        
        // Clear all dynamically created bounding boxes
        this.activeBoundBoxes.forEach((bboxInstance, key) => {
            if (bboxInstance) {
                bboxInstance.destroy();
            }
        });
        this.activeBoundBoxes.clear();
        
        print("Overlays cleared");
    }

    private cleanup(): void {
        if (this.microphoneProvider) {
            this.microphoneProvider.stop();
        }
        if (this.audioComponent && this.usePrerecordedAudio) {
            this.audioComponent.stop(false);
        }
        if (this.socket) {
            this.socket.close();
            this.socket = null;
        }
        
        // Clean up all bounding boxes
        this.clearAllOverlays();
    }
}
