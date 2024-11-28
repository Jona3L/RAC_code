import os
import subprocess
import sys
from xvfbwrapper import Xvfb

# Set environment variables
os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:/lib64:/lib"
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'  # Force software rendering

# Start Xvfb with specific screen settings
vdisplay = Xvfb(width=1024, height=768, colordepth=24)
vdisplay.start()

# Set DISPLAY variable explicitly
os.environ['DISPLAY'] = str(vdisplay.new_display)

print("Starting virtual display...")

# Print environment variables for debugging
print("Environment Variables:")
for key in ['PYOPENGL_PLATFORM', 'LD_LIBRARY_PATH', 'MESA_GL_VERSION_OVERRIDE', '__GLX_VENDOR_LIBRARY_NAME', '__NV_PRIME_RENDER_OFFLOAD']:
    print(f"{key}: {os.environ.get(key)}")

# Ensure PyOpenGL is installed
try:
    from OpenGL import GL
    print("OpenGL version:", GL.glGetString(GL.GL_VERSION))
except ImportError as e:
    print("Failed to import PyOpenGL:", e)
    sys.exit(1)

# Try different OpenGL backends
try:
    for platform in ['glx', 'egl']:
        os.environ['PYOPENGL_PLATFORM'] = platform
        print(f"\nTrying with PYOPENGL_PLATFORM={platform}")

        result = subprocess.run([sys.executable, "run_dialog.py"] + sys.argv[1:], 
                                check=False, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE,
                                universal_newlines=True,
                                env=os.environ)
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"Success with PYOPENGL_PLATFORM={platform}")
            sys.exit(0)
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    vdisplay.stop()

print("All attempts failed.")

# Diagnostic Information
print("\nDiagnostic Information:")
print("Python version:", sys.version)
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH'))
print("MESA_GL_VERSION_OVERRIDE:", os.environ.get('MESA_GL_VERSION_OVERRIDE'))
print("__GLX_VENDOR_LIBRARY_NAME:", os.environ.get('__GLX_VENDOR_LIBRARY_NAME'))
print("__NV_PRIME_RENDER_OFFLOAD:", os.environ.get('__NV_PRIME_RENDER_OFFLOAD'))
