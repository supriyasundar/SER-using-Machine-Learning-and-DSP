%%%%%%% SPEECH EMOTION RECOGNITION %%%%%%%%

clc;

close all;

clear all;

warning off;

rng('default');
    
%%%%%%CASE-1 BY USING MFCC,PLP,LPC %%%%%%%

[f,p]=uigetfile('*.wav');

[s,fs]= audioread([p,f]);

plot(s),title('INPUT VOICE SIGNAL');

%%%%%TO APPLY MFCC ALGORITHM %%%%%%%

sp=s;
% The function computed the mfcc output of the input signal s sampled at fs
%    s:    Samples
%    fs:   Sampling rate
%fs=100;
N=0.02*fs  ;                    % size of each frame
%M=156;
M=0.01*fs;                      % overlap size
nof=40;                     % number of filters
len=floor(length(s)/(N-M)) ;                    % The number of times for loop is to be run                                         ;
a(1:N,1:len)=0;             % framing the signal with overlap

% initialization of the first chunk
a(:,1)=s(1:N);
x=N-M;
 h= hamming(N);

for j=2:len-1
 

a(:,j)=s(x*(j-2)+1:x*(j-2)+N);

%    a(:,j)=s(x*j-1:x*j+N);
    end;
   
    for j=1:len;
   b(:,j)= a(:,j).* h;
end
% computes the mel filter bank coeffs
 %m=melfilterbank(nof,N,fs);         % normailising to mel freq
% The computation of the cepstrum coefficients
% The computation of the cepstrum coefficients
%m=melfilterbank(nof,N,fs); 
m=melfilterbank(nof,N,fs); 

for j=1:len-1
    y(:,j)=fft(b(:,j));            % calculating fft
    n2 = 1 + floor(N/2);        % adjust the dimensions of the vector y for mel filter banks
    ms = m * abs(y(1:n2,j)).^2;  % applies the mel filter bank
    v(:,j)=dct(log(ms));                  % converting back to time domain
end
v(1,:)=[];


% mfcc=mfcc_test2(s,fs);

mfcc=v;

size (mfcc)

mt=transpose(mfcc);


%%%%%%% TO TAKE MEAN VALUES %%%%%%

mt1=mean(mt);


%%%%%%% TAKE PERCEPTUAL LINEAR PREDICTING %%%%%%%%

PLP=plp(s,fs);

PLP=imresize(PLP,[1 280]);


%%%%%TAKE MEAN VALUE OF PLP %%%%%%

PLP=mean(PLP);

%%%%%%% NOW TAKE LINEAR PREDICTIVE CODING VALUES %%%%%%%%

frame=.02*fs;

   x1 = s.*hamming(length(s));
   
   preemph = [1 0.63];
   
   x1 = filter(1,preemph,x1);
   
 %%%%%%% FINAL LPC VALUES%%%%%%%
   
  B = lpc(x1,8);
  
 %%%%% NOW COMBINE THE  FEATURES OF MFCC,LPC,PLP %%%%%%
   
 tot=[mt1 PLP B];

%%%%%% LOAD THE FEATURES %%%%%


load feature.mat

%%%%%%CLASSIFICATION %%%%%%%%

label=ones(1,528);
label(131:212)=2;
label(213:258)=3;
label(259:325)=4;
label(326:392)=5;
label(393:469)=6;
label(470:528)=7;

model=fitcknn(feat,label);
result=predict(model,tot);

if result==1
    disp('Anger voice');
    msgbox('Anger voice');
elseif result==2
    disp('Boredom voice');
    msgbox('Boredom voice');
elseif result==3
    disp('Disgust voice');
    msgbox('Disgust voice');
elseif result==4
    disp('Fear voice');
    msgbox('Fear voice');
elseif result==5
    disp('Happyness voice');
    msgbox('Happyness voice');
elseif result==6 
    disp('Neutral voice');
    msgbox('Neutral voice');
elseif result==7
    disp('sadness voice');
    msgbox('sadness voice');
end




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
