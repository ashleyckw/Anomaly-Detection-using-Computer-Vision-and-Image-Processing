# Anomaly Detection for Image Processing (Django Project)

## Description

This Django project focuses on detecting anomalies within images using advanced image processing techniques. It provides a platform for users to upload, process, and view the results of anomaly detection on their images.

## Directory Structure

- **anomaly_detection/**: Main project configuration and settings.
- **image_input/**: Django app responsible for handling image uploads and processing.
- **media/**: Directory where uploaded images are stored.
- **static/**: Contains static files for the project.
- **db.sqlite3**: SQLite database file.
- **manage.py**: Django's command-line utility for administrative tasks.

## Installation & Setup

1. **Navigate to the project directory**:
   ```bash
   cd anomaly_detection
   ```

2. **Install the required packages**:
   ```bash
   pip install -r requirement_file.txt
   ```

3. **Migrate the database**:
   ```bash
   python manage.py migrate
   ```

4. **Run the development server**:
   ```bash
   python manage.py runserver
   ```

Visit `http://127.0.0.1:8000/` in your browser to access the application.

## Usage

1. **Uploading Images**: Navigate to the image upload section and upload the images you want to process.
2. **Image Registration or Alignment**: Perform feature detection, image matching, transformation calculation and image alignment to image batches.
3. **Morphological Methods**: Apply methods like greyscaling, filters, erosion, dilation onto the image batches.
4. **Anomaly Detection**: After processing, the results with highlighted anomalies (if detected) will be displayed.
5. **Visualization**: Addiitonal visualizations from all the previous steps will be presented in this section.