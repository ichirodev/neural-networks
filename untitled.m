data = load('fasterRCNNVehicleTrainingData.mat');
vehicleDataset = data.vehicleTrainingData;

dataDir = fullfile(toolboxdir('vision'), 'visiondata');
vehicleDataset.imageFilename = fullfile(dataDir, vehicleDataset.imageFilename);

idx = floor(0.6 * height(vehicleDataset));
trainingData = vehicleDataset(1:idx,:);
testData = vehicleDataset(idx:end,:);

inputLayer = imageInputLayer([32 32 3]);

filterSize = [3 3];
numFilters = 32;

middleLayers = [
    convolution2dLayer(filterSize, numFilters, 'Padding', 1)
    reluLayer()
    convolution2dLayer(filterSize, numFilters, 'Padding', 1)
    reluLayer()
    maxPooling2dLayer(3, 'Stride', 2)
];

finalLayers = [
    fullyConnectedLayer(64)
    reluLayer()
    fullyConnectedLayer(width(vehicleDataset))
    softmaxLayer()
    classificationLayer()
];

layers = [
    inputLayer
    middleLayers
    finalLayers
];

optionsStage1 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir ...
    );

optionsStage2 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir ...
    );

optionsStage3 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 256, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir ...
    );

optionsStage4 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir ...
    );

options = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];

doTrainingAndEval = false;

if doTrainingAndEval == true
    rng(0);

    detector = trainFasterRCNNObjectDetector( ...
        trainingData, layers, options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.6 1], ...
        'BoxPyramidScale', 1.2);
else
    detector = data.detector;
end

myCamera = webcam;

while true
    I = imresize(myCamera.snapshot, [1366,768]);
    
    [bboxes, scores] = detect(detector, I);
   
    if isempty(bboxes)
        titleImage = 'No car detected';
    else
        I = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
        titleImage = 'Car detected';
    end

    image(I)
    title(titleImage)
end