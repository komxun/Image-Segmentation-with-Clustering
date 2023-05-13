clc, clear, close all

path = 'C:\Users\Komsun\Desktop\Image-Segmentation-with-Clustering\data';

if exist(path, 'dir') == 7
   disp('Path exists')
else
   disp('Path does not exist !');
   disp("Please specify the path of 'data' folder")
end

addpath(path)
% Path for Testing, Validating, and Training images
imgTestDir  = [path, '\images\test\'];
imgValDir   = [path, '\images\val\'];
imgTrainDir = [path, '\images\train\'];

D_test   = dir(fullfile(imgTestDir,'*.jpg'));
D_val    = dir(fullfile(imgValDir,'*.jpg'));
D_train  = dir(fullfile(imgTrainDir,'*.jpg'));

% Path for Ground Truths 
gtTestDir = [path, '\groundTruth\test\'];
gtValDir  = [path, '\groundTruth\val\'];
gtTrainDir  = [path, '\groundTruth\train\'];


for i =1:numel(D_test)
    GT_test(i)  = load([gtTestDir D_test(i).name(1:end-4) '.mat']);
    imgTest = imread([imgTestDir D_test(i).name(1:end-4) '.jpg']);
    bndTest = double(GT_test(i).groundTruth{1}.Boundaries);  % use 1st GT for now
    segTest = double(GT_test(i).groundTruth{1}.Segmentation); % use 1st GT for now
    sz_test(i,:) = size(imgTest);
    
    % Reshape the data for simplicity
    IMG_test(:, i)   = reshape(imgTest, [], 1);
    GTbnd_test(:, i) = reshape(bndTest, [], 1);
    GTseg_test(:, i) = reshape(segTest, [], 1);
    GTclust_test(i,:) = length(unique(segTest));
end

%% displaying some of the imported images and ground ground truth

count = 1;
figure 
% for i = randi(100,[1,3]) %numel(D)
for i = 1:3
    subplot(2,3,count), imshow(reshape(IMG_test(:,i), sz_test(i,:)));
    subplot(2,3,count+3), imagesc(reshape(GTseg_test(:,i), sz_test(i,1:2)));
    count = count+1;
end

%% apply kmeans MATLAB function using uniform initial centers

num_eval = 5;

% For mean-shift clustering
bandWidth = 0.25;   % 20
plotFlag = false;

% For FCM
fuzziness = 2;

% ----------------Pre-process image filter Option-------------------
filterOpt = 2;

% Define a color range for segmentation
min_value = [0.2, 0.2, 0.2];
max_value = [0.8, 0.8, 0.8];

count = 1;
tic

numImgTest = 5;
% numImgTest = numel(D_test);

% Memory pre-allocation
precision_All = zeros(1,numImgTest);
recall_All = zeros(1,numImgTest);
F1Score_All = zeros(1,numImgTest);
jaccardIndex_All = zeros(1,numImgTest);
accuracy_All = zeros(1,numImgTest);
ssimIndex_All = zeros(1,numImgTest);

precision_All_ms = zeros(1,numImgTest);
recall_All_ms = zeros(1,numImgTest);
F1Score_All_ms = zeros(1,numImgTest);
jaccardIndex_All_ms = zeros(1,numImgTest);
accuracy_All_ms = zeros(1,numImgTest);
ssimIndex_All_ms = zeros(1,numImgTest);

% for j = randi(100,[1,3]) %numel(D)
for j = 1:numImgTest
% for j = 3
% for j = 173  % ship
% for j = 7
% for j = 152 % Butter fly

    K = 3;

    test0 = reshape(IMG_test(:,j), sz_test(j,:));

    % Feature #5 : Color Mask
    % Convert the image to HSV color space
    img_hsv = rgb2hsv(test0);
    
    % Define a color range for segmentation
    min_value = [0.2, 0.2, 0.2];
    max_value = [0.8, 0.8, 0.8];

    % Create a mask for the selected color range
    mask = (img_hsv(:,:,1) >= min_value(1) & img_hsv(:,:,1) <= max_value(1)) & ...
           (img_hsv(:,:,2) >= min_value(2) & img_hsv(:,:,2) <= max_value(2)) & ...
           (img_hsv(:,:,3) >= min_value(3) & img_hsv(:,:,3) <= max_value(3));
    mask = reshape(double(mask), [], 1);

    
    % Pre-processing
    feat0 = preproc(test0, filterOpt, 'none');
    feat1 = preproc(test0, filterOpt, 'lab');
    feat2 = preproc(test0, filterOpt, 'hsv');
    feat3 = preproc(test0, filterOpt, 'gray');
    feat4 = preproc(test0, filterOpt, 'lin');

    edges = edge(feat3,'Canny');
    edges = reshape(double(edges), [], 1);

    feat0 = reshape(double(feat0), [], 3);
    feat1 = reshape(double(feat1), [], 3);
    feat2 = reshape(double(feat2), [], 3);
    feat3 = reshape(double(feat3), [], 1);
    feat4 = reshape(double(feat4), [], 3);
    
    test =  [feat1 feat2];
    allFeat = [feat0 feat1 feat2 feat3 feat4];

    Xf = test;

    % --- K-means clustering ---
    [clust, centers] = kmeans(Xf, K, 'Start','plus'); % initial centroids are Uniformly samples
    segment_mat    = reshape(clust, sz_test(j,1:2));
    segIMG_km = label2rgb(segment_mat, 'parula');

    % --- Mean-shifted clustering ---
    [clustCent,data2cluster,cluster2dataCell] = MeanShiftCluster(Xf',bandWidth,plotFlag);
    segment_mat_ms = reshape(data2cluster, sz_test(j,1:2));
    segIMG_ms = label2rgb(segment_mat_ms, 'parula');

    % --- Fuzzy c-means clustering ---
    options = [fuzziness 25 0.001 0]; 
    % NaN -> sets the fuzzy exponent to its default value of 2
    % Max iterations = 25.
    % improves by less than 0.001 between two consecutive iterations
    % 0 no display

    [centers_f,U] = fcm(Xf, K, options);
    reCreate = zeros(sz_test(j,1:2));
    maxU = max(U);
    for p = 1:size(U,1)
        idx = find(U(p,:) == maxU);
        reCreate(idx) = p;
    end
    segIMG_fcm = label2rgb(reCreate, 'parula');

    GTshow = label2rgb(GTseg_test(:,j), 'parula');       
    figure
    subplot(2,3,1), imshow(reshape(IMG_test(:,j), sz_test(j,:)));
    title('Original image', 'FontSize', 16)
    subplot(2,3,2), imshow(reshape(Xf(:,1:3), sz_test(j,:)))
    title('Pre-processed image in LAB color space', 'FontSize', 16)
    subplot(2,3,3), imshow(reshape(GTshow, sz_test(j,:)))
    title('Ground Truth', 'FontSize', 16)
    subplot(2,3,4), imshow(segIMG_km)
    title(['K-means++ clustering, K = ' num2str(K)], 'FontSize', 16)
    subplot(2,3,5), imshow(segIMG_fcm)
    title({['FCM clustering: ' num2str(K) ' clusters'], ...
        ['Fuzziness = ' num2str(fuzziness)]}, 'FontSize', 16)
    subplot(2,3,6), imshow(segIMG_ms)
    title({['Mean-shifted clustering: bandwidth = ' num2str(bandWidth)], ...
        ['Number of clusters = ' num2str(length(cluster2dataCell))]}, 'FontSize', 16)

    sgtitle(['Image ' num2str(j) ': ' D_test(j).name], 'FontSize', 20)

    count = count +1;

    % ---------------- Performance Evaluation -----------------------
    showBinar_fcm = 0;  % 1: show plot  0: don't
    showBinar_ms = 0;   % 1: show plot  0: don't

    [Binar, T] = binaryMatch(GTseg_test, segIMG_fcm, sz_test, j, showBinar_fcm);
    [Binar_ms, T_ms] = binaryMatch(GTseg_test, segIMG_ms, sz_test, j, showBinar_ms);

    precision_vec = zeros(1,size(Binar.seg,2));
    recall_vec = zeros(1,size(Binar.seg,2));
    accuracy_vec = zeros(1,size(Binar.seg,2));
    F1Score_vec = zeros(1,size(Binar.seg,2));
    jaccardIndex_vec = zeros(1,size(Binar.seg,2));
    ssimIndex_vec = zeros(1,size(Binar.seg,2));

    precision_vec_ms = zeros(1,size(Binar_ms.seg,2));
    recall_vec_ms = zeros(1,size(Binar_ms.seg,2));
    accuracy_vec_ms = zeros(1,size(Binar_ms.seg,2));
    F1Score_vec_ms = zeros(1,size(Binar_ms.seg,2));
    jaccardIndex_vec_ms = zeros(1,size(Binar_ms.seg,2));
    ssimIndex_vec_ms = zeros(1,size(Binar_ms.seg,2));

    for k = 1:size(Binar.seg,2)
        % FCM------------------------------------------------------
        segGT_binary = Binar.seg(k).GT;
        segIMG_binary = Binar.seg(k).IMG;
        [precision, recall, accuracy, F1Score, jaccardIndex, ssimIndex] ...
            = myEval(segGT_binary, segIMG_binary);
        precision_vec(k) = precision;
        recall_vec(k) = recall;
        accuracy_vec(k) = accuracy;
        F1Score_vec(k) = F1Score;
        jaccardIndex_vec(k) = jaccardIndex;
        ssimIndex_vec(k) = ssimIndex;
    end

    for k = 1:size(Binar_ms.seg, 2)
        % MS------------------------------------------------------
        segGT_binary_ms = Binar_ms.seg(k).GT;
        segIMG_binary_ms = Binar_ms.seg(k).IMG;
        [precision_ms, recall_ms, accuracy_ms, F1Score_ms, jaccardIndex_ms, ssimIndex_ms] ...
            = myEval(segGT_binary_ms, segIMG_binary_ms);
        precision_vec_ms(k) = precision_ms;
        recall_vec_ms(k) = recall_ms;
        accuracy_vec_ms(k) = accuracy_ms;
        F1Score_vec_ms(k) = F1Score_ms;
        jaccardIndex_vec_ms(k) = jaccardIndex_ms;
        ssimIndex_vec_ms(k) = ssimIndex_ms;
    end

    % FCM-------------------------------------------------------------
    disp(['-(FCM) Image ' num2str(j) ': ' D_test(j).name ' -----'])
    disp(['                 Recall: ' num2str(mean(recall_vec))]);
    disp(['              Precision: ' num2str(mean(precision_vec))]);
    disp(['               Accuracy: ' num2str(mean(accuracy_vec))]);
    disp(['               F1 score: ' num2str(mean(F1Score_vec))]);
    disp(['          Jaccard index: ' num2str(mean(jaccardIndex_vec))]);
    disp(['             SSIM index: ' num2str(mean(ssimIndex_vec))]);
    disp('--------------------------------')

    % MS-------------------------------------------------------------
    disp(['--(MS) Image ' num2str(j) ': ' D_test(j).name ' -----'])
    disp(['                 Recall: ' num2str(mean(recall_vec_ms))]);
    disp(['              Precision: ' num2str(mean(precision_vec_ms))]);
    disp(['               Accuracy: ' num2str(mean(accuracy_vec_ms))]);
    disp(['               F1 score: ' num2str(mean(F1Score_vec_ms))]);
    disp(['          Jaccard index: ' num2str(mean(jaccardIndex_vec_ms))]);
    disp(['             SSIM index: ' num2str(mean(ssimIndex_vec_ms))]);
    disp('--------------------------------')

    precision_All(j) = mean(precision_vec);
    recall_All(j) = mean(recall_vec);
    accuracy_All(j) = mean(accuracy_vec);
    F1Score_All(j) = mean(F1Score_vec);
    jaccardIndex_All(j) = mean(jaccardIndex_vec);
    ssimIndex_All(j) = mean(ssimIndex_vec);

    precision_All_ms(j) = mean(precision_vec_ms);
    recall_All_ms(j) = mean(recall_vec_ms);
    accuracy_All_ms(j) = mean(accuracy_vec_ms);
    F1Score_All_ms(j) = mean(F1Score_vec_ms);
    jaccardIndex_All_ms(j) = mean(jaccardIndex_vec_ms);
    ssimIndex_All_ms(j) = mean(ssimIndex_vec_ms);
    
end
toc
% FCM----------------------------------------------------------------
disp('===========  SUMMARY  ==========')
disp(['FCM clustering for ' num2str(numImgTest) ' images'])
disp(['             Avg Recall: ' num2str(mean(recall_All))]);
disp(['          Avg Precision: ' num2str(mean(precision_All))]);
disp(['           Avg Accuracy: ' num2str(mean(accuracy_All))]);
disp(['           Avg F1 score: ' num2str(mean(F1Score_All))]);
disp(['      Avg Jaccard index: ' num2str(mean(jaccardIndex_All))]);
disp(['         Avg SSIM index: ' num2str(mean(ssimIndex_All))]);
disp('================================')

% MS----------------------------------------------------------------
disp('===========  SUMMARY  ==========')
disp(['Mean Shift clustering for ' num2str(numImgTest) ' images'])
disp(['             Avg Recall: ' num2str(mean(recall_All_ms))]);
disp(['          Avg Precision: ' num2str(mean(precision_All_ms))]);
disp(['           Avg Accuracy: ' num2str(mean(accuracy_All_ms))]);
disp(['           Avg F1 score: ' num2str(mean(F1Score_All_ms))]);
disp(['      Avg Jaccard index: ' num2str(mean(jaccardIndex_All_ms))]);
disp(['         Avg SSIM index: ' num2str(mean(ssimIndex_All_ms))]);
disp('================================')


%% Evaluating Cluster (not image segmentation performance)
% evalclusters(Xf,clust,'CalinskiHarabasz')
% evalclusters(Xf,clust,'DaviesBouldin')

%% Evaluation function
function [precision, recall, accuracy, F1Score, jaccardIndex, ssimIndex] = myEval(segGT_binary, segIMG_binary)
    % Compute the number of true positives (TP), 
    % true negatives (TN), false positives (FP), false negatives (FN)
    TP = sum(sum(segGT_binary & segIMG_binary));
    TN = sum(sum(~segGT_binary & ~segIMG_binary));
    FP = sum(sum(~segGT_binary & segIMG_binary));
    FN = sum(sum(segGT_binary & ~segIMG_binary));
    
    % Compute the precision, recall, and accuracy
    recall = TP / (TP + FN);
    precision = TP / (TP + FP);
    accuracy = (TP + TN) / (TP + TN + FP + FN);
       
    % Compute the F1-Score (Dice similarity coefficient)
    F1Score = 2 * (precision * recall) / (precision + recall);
    
    % Compute the Jaccard index
    jaccardIndex = TP / (TP + FP + FN);

    % Compute SSIM index
    ssimIndex = ssim(uint8(segIMG_binary), uint8(segGT_binary));       
end

%% Preprocessing function
function test0 = preproc(test0, filterOpt, colorSpace)

    % ==================================================================
    % Gassian Option
    sigma = 2; % Define the standard deviation for the Gaussian filter
    kernelSize = 5; % Define the size of the Gaussian filter kernel
    % ==================================================================
    % Filtering
    switch filterOpt
        case 0
        case 1
            % Gaussian filter
            h = fspecial('gaussian', kernelSize, sigma);
            test0 = imfilter(test0, h);
        case 2
            % Disk filter
            h = fspecial('disk', 5);
            test0 = imfilter(test0, h);
        case 3
            test0 = imgaussfilt(test0, 5);
        case 4
            h = fspecial('sobel');
            test0 = imfilter(test0, h);
        case 5
            h = fspecial('prewitt');
            test0 = imfilter(test0, h);
        case 6
            h = fspecial('disk', 5);
            test0 = imfilter(test0, h);
            h2 = fspecial('sobel');
            test0 = imfilter(test0, h);
    end

    % Convert to other color space
    switch colorSpace
        case 'none'
        case 'hsv'
            test0 = rgb2hsv(test0);
        case 'lab'
            test0 = rgb2lab(test0);
        case 'gray'
            test0 = rgb2gray(test0);
        case 'lin'
            test0 = rgb2lin(test0);
    end
    
%     test0(:,:,1)=ones(sz_test(j, 1:2))*0;
    
    % Perform contrast stretching
    test0 = imadjust(test0, stretchlim(test0), [0 1]);

    % Preform image filling
    test0 = imfill(test0,"holes");
end


%% Cluster matching function
function [Binar, T] = binaryMatch(GTseg_test, segIMG, sz_test, numPic, showPlot)
    
    % GTseg_test: Ground Truth Image (n x m)
    % segIMG: Segmented Image (n x m x 3)
    % sz_test: size of the testing image (numPic x n x m x 3)
    % numPic : index of the current image

    segGT = reshape(GTseg_test(:,numPic), sz_test(numPic,1:2));  
    segIMG2 = rgb2gray(segIMG);
    idxSegGT  = unique(segGT);
    idxSegIMG = unique(segIMG2);
    
    colName = {};
    rowName = {};
    
    for i = 1:length(idxSegGT)
        for j = 1:length(idxSegIMG)
            % Pre-allocation;
            BW_gt  = zeros(sz_test(numPic,1:2));
            BW_img = zeros(sz_test(numPic,1:2));

            % Locate each unique segments
            idxYes_img = segIMG2 == idxSegIMG(j);
            idxYes_gt = segGT == idxSegGT(i);
            BW_gt(idxYes_gt) = 1;
            BW_img(idxYes_img) = 1;

            % Calculate similarity for all combination 
            % between GT segments and obtained segments
            mat(i,j) = jaccard(BW_gt, BW_img);
            
            if i == 1
                colName = [colName {['IMG_seg' num2str(j)]}];
            end
        end
        rowName = [rowName {['GT_seg' num2str(i)]}];
    end
    
    % Put all similarities into a table
    T = array2table(mat, 'VariableNames', colName,'RowNames', rowName);

    if showPlot
        figure
        sgtitle(['Image ' num2str(numPic) ': Binary matching (white = active segment)'], 'FontSize', 20)
    end

    similarThresh = 0.4;  % Keep segment with >= 40% similarity 
    for j = 1:length(idxSegIMG)
        % Pre-allocated the image
        BW_gt = zeros(sz_test(numPic,1:2));
        BW_img = zeros(sz_test(numPic,1:2));
    
        % Select segment pair with highest similarity OR >= 40% similarity
        idxMax = find( (mat(:,j)==max(mat(:,j))) | (mat(:,j) >= similarThresh));

        % If >1 segments obtained, combine them
        for k = 1:length(idxMax)
            dum = segGT == idxSegGT(idxMax(k));
            BW_gt = BW_gt + dum;
        end
        % Map the segment into the pre-allocated image
        idxYes2 = segIMG2 == idxSegIMG(j);
        BW_img(idxYes2) =1 ;
    
        if showPlot
            subplot(2, length(idxSegIMG),j)
            imshow(BW_gt)
            set(gca, 'FontSize', 14)
            
            if length(idxMax) > 1
                title({['Ground Truth: cluster ' num2str(j)],...
                    ['(' num2str(length(idxMax)) ' segments combined)']})
            else
                title(['Ground Truth: cluster ' num2str(j)])
            end
        
            subplot(2, length(idxSegIMG),j+length(idxSegIMG))
            imshow(BW_img)
            title(['Segmented image: cluster ' num2str(j)])
            set(gca, 'FontSize', 14)
        end

        Binar.seg(j).GT  = BW_gt;
        Binar.seg(j).IMG = BW_img;
    end
    
end
