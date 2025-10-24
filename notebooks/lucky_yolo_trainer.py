# lucky_yolo_trainer.py

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO

# --- STEP 3: TRAIN YOLO ON MEDICAL IMAGES ---

class CustomYOLOTrainer:
    """Train YOLO model with custom optimizations like layer freezing."""
    
    def __init__(self, model_size="yolov8n.pt", data_yaml="../data/yolo_dataset/data.yaml"):
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        """
        Load a pre-trained YOLO model with proper PyTorch compatibility.
        """
        print("\n" + "=" * 60)
        print(f"STEP 3: LOADING PRE-TRAINED YOLO MODEL ({self.model_size})")
        print("=" * 60)
        
        # Get PyTorch version
        torch_version = torch.__version__
        print(f"PyTorch version: {torch_version}")
        
        try:
            # CRITICAL FIX: Monkey-patch torch.load to disable weights_only
            # This is the most reliable fix for PyTorch 2.6+
            original_torch_load = torch.load
            
            def patched_load(*args, **kwargs):
                # Force weights_only=False
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            # Apply the patch
            torch.load = patched_load
            
            # Now load the model
            self.model = YOLO(self.model_size)
            
            # Restore original torch.load
            torch.load = original_torch_load
            
            print(f"Successfully loaded {self.model_size}")
            return self.model
            
        except Exception as e:
            error_str = str(e)
            print(f"CRITICAL ERROR: Model load failed.")
            print(f"   Error: {error_str}")
            
            # Try alternative approach: set environment variable before import
            print("\n   Attempting environment variable fix...")
            os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'
            
            # Force reimport of ultralytics with new env var
            if 'ultralytics' in sys.modules:
                # Try to reload ultralytics modules
                try:
                    import importlib
                    importlib.reload(sys.modules['ultralytics'])
                    from ultralytics import YOLO as YOLO_Reloaded
                    self.model = YOLO_Reloaded(self.model_size)
                    print(f"Successfully loaded with env var workaround")
                    return self.model
                except Exception as e2:
                    print(f"   Reload failed: {e2}")
            
            # Final fallback: Try to download and load fresh
            print("\n   Attempting fresh download...")
            try:
                # Clear any cached model
                cache_dir = Path.home() / '.cache' / 'torch' / 'hub' / 'ultralytics'
                if cache_dir.exists():
                    import shutil
                    shutil.rmtree(cache_dir, ignore_errors=True)
                
                self.model = YOLO(self.model_size)
                print(f"Successfully loaded after cache clear")
                return self.model
            except Exception as e3:
                print(f"   All attempts failed: {e3}")
                print("\n   MANUAL FIX REQUIRED:")
                print("   Option 1: Downgrade PyTorch: pip install torch==2.0.1 torchvision==0.15.2")
                print("   Option 2: Downgrade Ultralytics: pip install ultralytics==8.0.196")
                print("   Option 3: Run this before loading model:")
                print("            import torch")
                print("            torch.serialization.add_safe_globals(['ultralytics'])")
                
            self.model = None
            return None

    def freeze_backbone(self, num_layers=10):
        """
        Freezes the initial layers of the YOLOv8 model backbone for efficient fine-tuning.
        """
        if self.model is None:
            print("Error: Model not loaded. Cannot freeze layers.")
            return

        print("\n" + "=" * 60)
        print(f"FREEZING BACKBONE LAYERS (0 to {num_layers - 1})")
        print("=" * 60)
        
        try:
            # Access the actual PyTorch model
            model = self.model.model
            
            # Get the model layers - YOLOv8 stores layers in model.model attribute
            if hasattr(model, 'model') and hasattr(model.model, '__iter__'):
                model_layers = list(model.model)
            elif hasattr(model, 'children'):
                model_layers = list(model.children())
            else:
                print("Warning: Could not access model layers. Freezing all backbone parameters.")
                # Fallback: freeze by parameter names
                frozen_count = 0
                for name, param in model.named_parameters():
                    if 'model' in name and not any(x in name for x in ['head', 'detect', 'cv3']):
                        param.requires_grad = False
                        frozen_count += 1
                print(f"Froze {frozen_count} backbone parameters. Head remains trainable.")
                return
            
            frozen_count = 0
            for i, module in enumerate(model_layers):
                if i < num_layers:
                    for param in module.parameters():
                        param.requires_grad = False
                    frozen_count += 1
                else:
                    for param in module.parameters():
                        param.requires_grad = True
            
            print(f"Successfully froze {frozen_count} layers. Head remains trainable.")
            
        except Exception as e:
            print(f"Warning: Could not freeze layers: {e}")
            print("Continuing without layer freezing...")

    def train(self, epochs=50, img_size=640, batch_size=8, patience=10):
        """Train the model on medical images."""
        print("\n" + "=" * 60)
        print("STEP 4: TRAINING YOLO MODEL")
        print("=" * 60)
        
        if self.model is None:
            self.load_model() 
            
        if self.model is None:
            print("Training skipped: Model could not be loaded.")
            return None
            
        if not os.path.exists(self.data_yaml):
            print(f"Error: data.yaml not found at {self.data_yaml}")
            return None

        print(f"Training on: {self.device}")
        
        # CRITICAL: Apply torch.load patch for entire training process
        original_torch_load = torch.load
        
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_load
        
        try:
            results = self.model.train(
                data=self.data_yaml,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                patience=patience,
                save=True,
                device=self.device,
                augment=True,
                mosaic=0.5,
                mixup=0.0,
                copy_paste=0.0,
                hsv_h=0.01,
                hsv_s=0.3,
                hsv_v=0.2,
                degrees=5,
                translate=0.1,
                scale=0.2,
                flipud=0.5,
                fliplr=0.5,
                verbose=True
            )
            
            print("\nTraining complete!")
            return results
        
        except Exception as e:
            print(f"Error during training: {e}")
            return None
        
        finally:
            # Restore original torch.load after training
            torch.load = original_torch_load

    def validate(self):
        """Run validation on test set."""
        print("\n" + "=" * 60)
        print("STEP 5: VALIDATING DETECTION ACCURACY")
        print("=" * 60)
        if self.model is None:
            print("Error: No model loaded")
            return None
        
        # Apply torch.load patch for validation
        original_torch_load = torch.load
        
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = patched_load
        
        try:
            metrics = self.model.val(data=self.data_yaml, split='test')
            
            print(f"\nMetrics:")
            print(f" mAP50: {metrics.box.map50:.4f}")
            print(f" mAP50-95: {metrics.box.map:.4f}")
            print(f" Precision: {metrics.box.mp:.4f}")
            print(f" Recall: {metrics.box.mr:.4f}")
            
            return metrics
        
        except Exception as e:
            print(f"Error during validation: {e}")
            return None
        
        finally:
            # Restore original torch.load
            torch.load = original_torch_load