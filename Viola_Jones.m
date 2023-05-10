% Load dataset of faces and their corresponding bounding boxes
load('faces_roi.mat');

% Set if you want to train the model or just to detect a face
train = false;

if train
    % Retrieve faces from 'faces' variable loaded from faces_roi.mat
    positiveFaces = faces;

    % Load the negative faces file from the training set
    negativeFacesFile = fullfile('face_archive/real_and_fake_face/training_fake');
    
    % Create a name for the detector, and pass in the positive and negative
    % images, as well as the training options, which are outlined in the
    % paper on what they do.
    trainCascadeObjectDetector('face_detector.xml', positiveFaces, negativeFacesFile, ...
        'FalseAlarmRate', 0.01, ...
        'ObjectTrainingSize', [32 32], ...
        "NumCascadeStages", 15);
end

% Load the detector that we've previously trained
detector = vision.CascadeObjectDetector('face_detector.xml');

% Setup how we are going to play the video, and setup coordinates to
% position it on the screen
video = VideoReader("figures/figure_1.mp4");
videoPlayer  = vision.VideoPlayer("Position", [350 100 700 700]);

% Loop over each frame in video
while hasFrame(video)
    % Retrieve the next frame of the video
    nextFrame = readFrame(video);

    % Retrieve the bounding box for any faces in the current frame image
    boundingBox = step(detector, nextFrame);

    % Insert the bounding box to the frame
    nextFrame = insertObjectAnnotation(nextFrame, "rectangle", boundingBox, 'Face');

    % Display the frame then move on to the next frame
    step(videoPlayer, nextFrame);
end