%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% speech emotion recognition %%%%%%%%%%%%

clc
close all
clear all
warning off;
rng('default');
feat=[];
s1=dir('Anger\*.wav');
for ii=1:length(s1)
    b = fullfile('Anger',s1(ii).name);
    
%%

% [f,p]=uigetfile('*.wav');
[s,fs]= audioread([b]);
% [s,fs]= wavread(strcat('1 ('));
mfcc=mfcc_test2(s,fs);
PLP=plp(s,fs);
PLP=imresize(PLP,[1 280]);
PLP=mean(PLP);
size (mfcc)
mt=transpose(mfcc);
mt1=mean(mt);

mt11=mt1(:,1);
mt22=mt1(:,22);

mtot=[mt11];

frame=.02*fs;
   x1 = s.*hamming(length(s));
   preemph = [1 0.63];
   x1 = filter(1,preemph,x1);
   B = lpc(x1,8);
mt2=[mtot];
feat=[feat;mt2]
end
s1=dir('Boredom\*.wav');
for ii=1:length(s1)
    b = fullfile('Boredom',s1(ii).name);
    
%%

% [f,p]=uigetfile('*.wav');
[s,fs]= audioread([b]);
% [s,fs]= wavread(strcat('1 ('));
mfcc=mfcc_test2(s,fs);
PLP=plp(s,fs);
PLP=imresize(PLP,[1 280]);
PLP=mean(PLP);
size (mfcc)
mt=transpose(mfcc);
mt1=mean(mt);

mt11=mt1(:,1);
mt22=mt1(:,22);

mtot=[mt11];

frame=.02*fs;
   x1 = s.*hamming(length(s));
   preemph = [1 0.63];
   x1 = filter(1,preemph,x1);
   B = lpc(x1,8);
mt2=[mtot];
feat=[feat;mt2]
end
s1=dir('Disgust\*.wav');
for ii=1:length(s1)
    b = fullfile('Disgust',s1(ii).name);
    
%%

% [f,p]=uigetfile('*.wav');
[s,fs]= audioread([b]);
% [s,fs]= wavread(strcat('1 ('));
mfcc=mfcc_test2(s,fs);
PLP=plp(s,fs);
PLP=imresize(PLP,[1 280]);
PLP=mean(PLP);
size (mfcc)
mt=transpose(mfcc);
mt1=mean(mt);

mt11=mt1(:,1);
mt22=mt1(:,22);

mtot=[mt11];

frame=.02*fs;
   x1 = s.*hamming(length(s));
   preemph = [1 0.63];
   x1 = filter(1,preemph,x1);
   B = lpc(x1,8);
mt2=[mtot];
feat=[feat;mt2]
end
s1=dir('Fear\*.wav');
for ii=1:length(s1)
    b = fullfile('Fear',s1(ii).name);
    
%%

% [f,p]=uigetfile('*.wav');
[s,fs]= audioread([b]);
% [s,fs]= wavread(strcat('1 ('));
mfcc=mfcc_test2(s,fs);
PLP=plp(s,fs);
PLP=imresize(PLP,[1 280]);
PLP=mean(PLP);
size (mfcc)
mt=transpose(mfcc);
mt1=mean(mt);

mt11=mt1(:,1);
mt22=mt1(:,22);

mtot=[mt11];

frame=.02*fs;
   x1 = s.*hamming(length(s));
   preemph = [1 0.63];
   x1 = filter(1,preemph,x1);
   B = lpc(x1,8);
mt2=[mtot];
feat=[feat;mt2]
end
s1=dir('Happyness\*.wav');
for ii=1:length(s1)
    b = fullfile('Happyness',s1(ii).name);
    
%%

% [f,p]=uigetfile('*.wav');
[s,fs]= audioread([b]);
% [s,fs]= wavread(strcat('1 ('));
mfcc=mfcc_test2(s,fs);
PLP=plp(s,fs);
PLP=imresize(PLP,[1 280]);
PLP=mean(PLP);
size (mfcc)
mt=transpose(mfcc);
mt1=mean(mt);

mt11=mt1(:,1);
mt22=mt1(:,22);

mtot=[mt11];

frame=.02*fs;
   x1 = s.*hamming(length(s));
   preemph = [1 0.63];
   x1 = filter(1,preemph,x1);
   B = lpc(x1,8);
mt2=[mtot];
feat=[feat;mt2]
end
s1=dir('Neutral\*.wav');
for ii=1:length(s1)
    b = fullfile('Neutral',s1(ii).name);
    
%%

% [f,p]=uigetfile('*.wav');
[s,fs]= audioread([b]);
% [s,fs]= wavread(strcat('1 ('));
mfcc=mfcc_test2(s,fs);
PLP=plp(s,fs);
PLP=imresize(PLP,[1 280]);
PLP=mean(PLP);
size (mfcc)
mt=transpose(mfcc);
mt1=mean(mt);
mt11=mt1(:,1);
mt22=mt1(:,22);

mtot=[mt11];
frame=.02*fs;
   x1 = s.*hamming(length(s));
   preemph = [1 0.63];
   x1 = filter(1,preemph,x1);
   B = lpc(x1,8);
mt2=[mtot];
feat=[feat;mt2]
end
s1=dir('Sadness\*.wav');
for ii=1:length(s1)
    b = fullfile('Sadness',s1(ii).name);
    
%%

% [f,p]=uigetfile('*.wav');
[s,fs]= audioread([b]);
% [s,fs]= wavread(strcat('1 ('));
mfcc=mfcc_test2(s,fs);
PLP=plp(s,fs);
PLP=imresize(PLP,[1 280]);
PLP=mean(PLP);
size (mfcc)
mt=transpose(mfcc);
mt1=mean(mt);

mt11=mt1(:,1);
mt22=mt1(:,22);

mtot=[mt11];

frame=.02*fs;
   x1 = s.*hamming(length(s));
   preemph = [1 0.63];
   x1 = filter(1,preemph,x1);
   B = lpc(x1,8);
mt2=[mtot];
feat=[feat;mt2]
end

save 'feature2.mat' feat


