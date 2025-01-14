# Final-Project-Avalon

## Setup

To create a virtual environment and install the dependencies, follow these steps:

1. Create a virtual environment:
    ```sh
    python3 -m venv env
    ```

2. Activate the virtual environment:
    ```sh
    source env/bin/activate
    ```

    ```
    .\env\Scripts\activate

    #if using GPU
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```