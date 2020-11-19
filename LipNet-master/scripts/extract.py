import dlib
import cv2
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-sp", "--source-path", required=True, help="source directory of the videos")
ap.add_argument("-op", "--output-path", required=True, help="source directory of the final output")
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-l", "--log-path", required=True, help="path to log errors")

args = vars(ap.parse_args())

VIDEO_PATH = args["source_path"]   # Dataset path
TARGET_PATH  = args["output_path"] # The path that the result images will be saved
SHAPE_PREDICTOR_PATH = args["shape_predictor"] # Shape predictor path
LOG_PATH = args["log_path"]        # The path for the working log file           
LIP_MARGIN = 0.5                # Marginal rate for lip-only image.
RESIZE = (100,50)               # Final image size

logfile = open(LOG_PATH,'w')

# Face detector and landmark detector
face_detector = dlib.get_frontal_face_detector()   
landmark_detector = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

def find_files(directory):                              # Read video list
    for root, dirs, files in os.walk(directory):
        for basename in files:
            filename = os.path.join(root, basename)
            yield filename

def shape_to_list(shape):
	coords = []
	for i in range(0, 68):
		coords.append((shape.part(i).x, shape.part(i).y))
	return coords

def obtain_face_landmark(frame_buffer):
    landmark_buffer = []                                # A list to hold face landmark information
    for (i, image) in enumerate(frame_buffer):          # Iterate on frame buffer
        face_rects = face_detector(image,1)             # Detect face
        
        if len(face_rects) < 1:                         # No face detected
            print("No face detected: ",vid_path)
            logfile.write(vid_path + " : No face detected \r\n")
            break

        rect = face_rects[0]                            # Proper number of face
        landmark = landmark_detector(image, rect)       # Detect face landmarks
        landmark = shape_to_list(landmark)
        landmark_buffer.append(landmark)
    return landmark_buffer    

for vid_name in find_files(VIDEO_PATH):                 # Iterate on video files
    print(f"Processing {vid_name}")
    vid = cv2.VideoCapture(vid_name)                    # Read video

    # Parse into frames 
    frame_buffer = []                                   # A list to hold frame images
    frame_buffer_color = []                             # A list to hold original frame images
    frame_num = 0
    while(True):
        success, frame = vid.read()                     # Read frame
        if not success:
            break                                       # Break if no frame to read left
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   # Convert image into grayscale
        frame_buffer.append(gray)                       # Add image to the frame buffer
        frame_buffer_color.append(frame)

    vid.release()

    # Obtain face landmark information
    landmark_buffer = obtain_face_landmark(frame_buffer)

    # Crop images
    cropped_buffer = []
    for (i,landmark) in enumerate(landmark_buffer):
        lip_landmark = landmark[48:68]                                          # Landmark corresponding to lip
        lip_x = sorted(lip_landmark,key = lambda pointx: pointx[0])             # Lip landmark sorted for determining lip region
        lip_y = sorted(lip_landmark, key = lambda pointy: pointy[1])
        x_add = int((-lip_x[0][0]+lip_x[-1][0])*LIP_MARGIN)                     # Determine Margins for lip-only image
        y_add = int((-lip_y[0][1]+lip_y[-1][1])*LIP_MARGIN)
        crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add, lip_y[0][1]-y_add, lip_y[-1][1]+y_add)   # Crop image
        cropped = frame_buffer_color[i][crop_pos[2]:crop_pos[3],crop_pos[0]:crop_pos[1]]
        cropped = cv2.resize(cropped,(RESIZE[0],RESIZE[1]),interpolation=cv2.INTER_CUBIC)        # Resize
        cropped_buffer.append(cropped)

    # Save result
    filepath = os.path.splitext(vid_name)[0]
    path = os.path.normpath(filepath)
    path = path.split(os.sep)
    target_dir = os.path.join(TARGET_PATH, path[-2], path[-1])
    
    for (i,image) in enumerate(cropped_buffer):
        if not os.path.exists(target_dir):           # If the directory not exists, make it.
            os.makedirs(target_dir)
        cv2.imwrite(os.path.join(target_dir, "mouth_{0:03d}.png".format(i)), image)     # Write lip image

logfile.close()
