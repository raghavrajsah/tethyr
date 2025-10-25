@component
export class ARServerCommunication extends BaseScriptComponent {
    // Modules
    private cameraModule: CameraModule = require('LensStudio:CameraModule');
    private internetModule: InternetModule = require('LensStudio:InternetModule');
    
    // Camera related
    private cameraRequest: CameraModule.CameraRequest;
    private cameraTexture: Texture;
    private cameraTextureProvider: CameraTextureProvider;
    
    // Server configuration
    @input
    @hint('Your localhost server URL, e.g., http://localhost:5000')
    serverUrl: string = 'http://localhost:5000';
    
    @input
    @hint('Visual marker to position at returned coordinates')
    labelMarker: SceneObject | undefined;
    
    @input
    @hint('Text component to display label')
    labelText: Text | undefined;
    
    // Frame processing control
    private isProcessingFrame: boolean = false;
    private frameCounter: number = 0;
    
    @input
    @hint('Process every Nth frame (e.g., 30 = once per second at 30fps)')
    frameSkip: number = 60;

    onAwake() {
        this.createEvent('OnStartEvent').bind(() => {
            this.initializeCamera();
        });
    }
    
    private initializeCamera(): void {
        // Create camera request
        this.cameraRequest = CameraModule.createCameraRequest();
        this.cameraRequest.cameraId = CameraModule.CameraId.Default_Color;
        
        // Request camera access
        this.cameraTexture = this.cameraModule.requestCamera(this.cameraRequest);
        this.cameraTextureProvider = this.cameraTexture.control as CameraTextureProvider;
        
        // Set up frame callback
        this.cameraTextureProvider.onNewFrame.add((cameraFrame) => {
            this.onNewCameraFrame(cameraFrame);
        });
        
        print('Camera initialized successfully');
    }
    
    private onNewCameraFrame(cameraFrame: CameraFrame): void {
        // Skip frames to avoid overwhelming the server
        this.frameCounter++;
        if (this.frameCounter % this.frameSkip !== 0) {
            return;
        }
        
        // Avoid concurrent processing
        if (this.isProcessingFrame) {
            return;
        }
        
        this.isProcessingFrame = true;
        this.processFrame();
    }
    
    private processFrame(): void {
        // Encode texture to base64
        Base64.encodeTextureAsync(
            this.cameraTexture,
            (encodedString: string) => {
                // Success callback
                this.sendToServer(encodedString);
            },
            () => {
                // Failure callback
                print('Failed to encode texture');
                this.isProcessingFrame = false;
            },
            CompressionQuality.HighQuality,
            EncodingType.Jpg
        );
    }
    
    private async sendToServer(base64Image: string): Promise<void> {
        try {
            // Create POST request
            const request = new Request(`${this.serverUrl}/process`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: base64Image,
                    timestamp: Date.now()
                })
            });
            
            const response = await this.internetModule.fetch(request);
            
            if (response.status === 200) {
                const data = await response.json();
                // const data = responseData.json;
                
                print(`Server response: x=${data.x}, y=${data.y}, label=${data.label}`);
                
                // Display the label at the coordinates
                this.displayLabel(data.x, data.y, data.label);
            } else {
                print(`Server returned status ${response.status}`);
            }
        } catch (error) {
            print(`Error sending to server: ${error}`);
        } finally {
            this.isProcessingFrame = false;
        }
    }
    
    private displayLabel(x: number, y: number, label: string): void {
        // Update text label if available
        if (this.labelText) {
            this.labelText.text = label;
        }
        
        // Position the marker at the specified coordinates
        if (this.labelMarker) {
            // Get camera information
            const camera = global.deviceInfoSystem.getTrackingCameraForId(
                CameraModule.CameraId.Default_Color
            );
            
            // Normalize coordinates (assuming server returns pixel coordinates)
            const resolution = camera.resolution;
            const normalizedX = x / resolution.x;
            const normalizedY = y / resolution.y;
            
            // Convert normalized 2D coordinates to 3D world position
            // Depth is in centimeters from the camera
            const depth = 100;
            const worldPos = camera.unproject(new vec2(normalizedX, normalizedY), depth);
            
            this.labelMarker.getTransform().setWorldPosition(worldPos);
            
            print(`Marker positioned at world coordinates: ${worldPos}`);
        }
    }
}