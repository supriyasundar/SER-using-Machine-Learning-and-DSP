% % %%%
clc;
clear;
close all;
warning off
% % %% dataset images

[f,p]=uigetfile('*.wav');
[s,fs]= audioread([p,f]);
plot(s);
title('input');
xlabel('time');
ylabel('amplitude');



ff=fft(s);
plot(real(ff));
ff=real(ff);

%%%convert signal into image%%%%%
Fr=getframe;
figure,imshow(Fr.cdata);
T=Fr.cdata;
I=imadjust(T,[]);
figure,imshow(I);
% % 
I=imresize(I,[100 100]);
figure,imshow(I );
I=imadjust(I,[]);
figure,imshow(I);

% % % %%%%%%%%%%% CONVERT THE DATA TYPE INTO UNSIGNED INTEGER %%%%%%%%%%%
re=im2uint8(I);




% % % %%%%% TRAIN THE DATASET IMAGES %%%%%
% % %  
% % % matlabroot='C:\Users\SPIRO-IMAGEPROCESSIN\Desktop\test\ITIMP42-DEEP_SPEECH_FINISHED\train';
% % % 
% % % Data=imageDatastore(matlabroot,'IncludeSubfolders',true,'LabelSource','foldernames');
% % % 
% % % %%% CREATE CONVOLUTIONAL NEURAL NETWORK LAYERS %%%%%%
% % % 
% % % 
% % % layers=[imageInputLayer([100 100 3])  
% % %     
% % % convolution2dLayer(3,8,'Padding','same')
% % %     batchNormalizationLayer
% % %     reluLayer
% % %     
% % %     maxPooling2dLayer(2,'Stride',2)
% % %     
% % %     convolution2dLayer(3,16,'Padding','same')
% % %     batchNormalizationLayer
% % %     reluLayer
% % %     
% % %     maxPooling2dLayer(2,'Stride',2)
% % %     
% % %     convolution2dLayer(3,32,'Padding','same')
% % %     batchNormalizationLayer
% % %     reluLayer
% % %     
% % %     maxPooling2dLayer(2,'Stride',2)
% % %     
% % %      convolution2dLayer(3,64,'Padding','same')
% % %     batchNormalizationLayer
% % %     reluLayer
% % %     
% % %     maxPooling2dLayer(2,'Stride',2)
% % %     
% % %      convolution2dLayer(3,128,'Padding','same')
% % %     batchNormalizationLayer
% % %     reluLayer
% % %     
% % %     maxPooling2dLayer(2,'Stride',2)
% % %     
% % %     
% % %     convolution2dLayer(3,256,'Padding','same')
% % %     batchNormalizationLayer
% % %     reluLayer
% % %     
% % %     maxPooling2dLayer(2,'Stride',2)
% % %     
% % %     fullyConnectedLayer(7)
% % %     softmaxLayer
% % %     classificationLayer];
% % % 
% % % options=trainingOptions('sgdm','MaxEpochs',15,'initialLearnRate',0.01,'Plots','training-progress');
% % % 
% % % convnet=trainNetwork(Data,layers,options);
% % % 
% % % 
% % % save convnet.mat convnet
% % % % % % % 
load convnet.mat





layer = 'conv_1';

channels = 1:8;
% 
I = deepDreamImage(convnet,layer,channels, ...
    'PyramidLevels',1, ...
    'Verbose',0);


 figure
for i = 1:8
    
    subplot(4,2,i)
   imshow(I(:,:,:,i))
end


imgSize = size(re);
imgSize = imgSize(1:2);

act1 = activations(convnet,re,'conv_1','OutputAs','channels');

sz = size(act1);

act1 = reshape(act1,[sz(1) sz(2) 1 sz(3)]);
figure
montage(mat2gray(act1),'Size',[8 12])


act1ch32 = act1(:,:,:,8);

act1ch32 = mat2gray(act1ch32);

act1ch32 = imresize(act1ch32,imgSize);
figure
imshowpair(re,act1ch32,'montage')
% % 

layer = 'conv_1';
channels = 1:8;

I = deepDreamImage(convnet,layer,channels, ...
    'PyramidLevels',1, ...
    'Verbose',0);


 figure
for i = 1:8
    
    subplot(4,2,i)
   imshow(I(:,:,:,i))
end
%%%%% TO  CLASSIFY THE OUTPUT %%%%%%% 

output=classify(convnet,re);
tf1=[];

for ii=1:7
    st=int2str(ii)
    tf=ismember(output,st);
    tf1=[tf1 tf]
end

output1=find(tf1==1);

if output1==1

   msgbox('ANGER');

elseif output1==2

    msgbox('BOREDOM');
    
    
    
    elseif output1==3

    msgbox('DISGUST');
    
    elseif output1==4

    msgbox('FEAR');
    
    elseif output1==5

    msgbox('HAPPYNESS');
    
    elseif output1==6

    msgbox('NEUTRAL');
    
    elseif output1==7

    msgbox('SADNESS');
        
end     
%%%%%%%%%%%%%%%%%%%%%%%end%%%%%%% 
