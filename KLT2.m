detector = vision.CascadeObjectDetector();

% Load the video file of a face moving
video = VideoReader("my_face.mp4");

% Load the initial frame of the video
initialFrame = readFrame(video);

% Generate the bounding box detected from the cascade object detector
boundingBox = step(detector, initialFrame);

% Need to make sure bounding box is shown in initial frame image
initialFrame = insertShape(initialFrame, "rectangle", boundingBox);

% Show the initial frame image
figure;
imshow(initialFrame);
title("A face has been detected!");

% Convert the bounding box into an array of 4 points
boundingBoxPoints = bbox2points(boundingBox(1, :));

% Identify the unqiue feature points in the initial frame
points = detectMinEigenFeatures(im2gray(initialFrame), "ROI", boundingBox);

% Display the new frame with the tracked points
figure,
imshow(initialFrame),
hold on,
title("Detected unique features of the identified face!");
plot(points);

% Initialize a points tracker to keep track of points throughout
% video frames
pointTracker = vision.PointTracker("MaxBidirectionalError", 2);

% Load the tracker with the points that have already been detected
points = points.Location;
initialize(pointTracker, points, initialFrame);

% Load the video player
videoPlayer  = vision.VideoPlayer("Position", [350 100 700 700]);
oldPoints = points;

while hasFrame(video)
    % get the next frame
    nextFrame = readFrame(video);

    % Identify the points after the frame has changed
    [points, isPointFound] = step(pointTracker, nextFrame);
    newFramePoints = points(isPointFound, :);
    oldFramePoints = oldPoints(isPointFound, :);
    
    % Detect the transformation between the old points and the new points,
    % if there is less than two points, it has not worked and points will
    % likely be lost in the next frame
    if size(newFramePoints, 1) >= 2
        
        [form, x] = estimateGeometricTransform2D(oldFramePoints, newFramePoints, "similarity", "MaxDistance", 4);
        oldFramePoints    = oldFramePoints(x, :);
        newFramePoints = newFramePoints(x, :);
        
        % Apply the new points to the frame
        bboxPoints = transformPointsForward(form, boundingBoxPoints);
                
        % Ensure bounding box stays around object as video keeps playing
        boundingBoxPoly = reshape(boundingBoxPoints', 1, []);
        nextFrame = insertShape(nextFrame, "polygon", boundingBoxPoly, "LineWidth", 2);
                
        % Display the transformed tracked points
        nextFrame = insertMarker(nextFrame, newFramePoints, "+", "Color", "white");       
        
        % Delete the old points and replace them with the newly transformed
        % ones
        oldPoints = newFramePoints;
        setPoints(pointTracker, oldPoints);        
    end
    
    % Show the next frame
    step(videoPlayer, nextFrame);
end

release(videoPlayer);
release(pointTracker);