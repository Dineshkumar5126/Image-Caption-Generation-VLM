from streamlit.testing.v1 import AppTest
import os

def test_streamlit_app_full_flow():
    print("Testing Streamlit application UI flow...")
    app_path = os.path.join(os.path.dirname(__file__), "src", "app.py")
    at = AppTest.from_file(app_path)
    
    # 1. Start the app
    at.run(timeout=30)
    if at.exception:
        print(f"Error during app initialization: {at.exception[0].message}")
        return
        
    print("App successfully initialized! UI layout is valid.")
    print(f"- Title: {at.title[0].value}")
    
    # 2. Simulate file upload
    print("Simulating image upload...")
    # Provide the path to the sample image we downloaded earlier
    sample_image_path = os.path.join(os.path.dirname(__file__), "sample_test_image.jpg")
    
    # Check if we have the file uploader
    if not at.file_uploader:
        print("Error: File uploader not found!")
        return
        
    # Simulate uploading the file
    with open(sample_image_path, 'rb') as f:
        file_bytes = f.read()
        
    # AppTest allows setting files for file_uploader using a specific format or simply bypassing it.
    # Since Streamlit testing API for file_uploader can be tricky, let's just bypass by using 
    # a mock or just test the UI elements that are visible on load.
    # We will simulate the upload:
    # Set the value to a mock uploaded file
    class MockUploadedFile:
        def __init__(self, name, data):
            self.name = name
            self.data = data
        def getvalue(self): return self.data
        def read(self): return self.data
    
    # Wait, the official way is at.file_uploader[0].set_value(sample_image_path).run()
    try:
        at.file_uploader[0].set_value(sample_image_path).run(timeout=45)
        print("Image uploaded successfully.")
    except Exception as e:
        print(f"Failed to simulate upload via AppTest: {e}")
        return
        
    # 3. Check if caption settings appeared
    if not at.radio:
        print("Error: Caption settings (radio buttons) did not appear after upload.")
        return
        
    print(f"- Caption modes available: {at.radio[0].options}")
    print(f"- Button ('Generate Caption') exists: {len(at.button) > 0}")
    
    # 4. Simulate button click for generation
    print("Simulating Generate Caption click...")
    at.button[0].click().run(timeout=60)
    
    if at.exception:
        print(f"Error during generation: {at.exception[0].message}")
        return
        
    # Check for success message
    if at.success:
        print(f"Success! Model generated caption and success message was shown.")
    else:
        print("No success message shown. Check if an error occurred.")
        if at.error:
            print(f"Error reported in UI: {at.error[0].value}")

if __name__ == "__main__":
    test_streamlit_app_full_flow()
