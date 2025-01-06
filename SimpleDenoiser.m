classdef SimpleDenoiser < audioPlugin

    properties

        Bypass = false;
        %MODELSELECT='CNN';
        TrainingAudio = 'Washing Machine';
        Strength = 1;

    end

    properties (Constant)

        PluginInterface = audioPluginInterface(...
            audioPluginParameter('Bypass', 'DisplayName', 'Bypass', 'Mapping', {'enum', 'Off', 'On'}), ...
            audioPluginParameter('TrainingAudio', 'DisplayName', 'Training Audio','Mapping', {'enum', 'Washing Machine', 'White Noise'}), ...
            audioPluginParameter('Strength', 'DisplayName', 'Strength', 'Mapping', {'lin', 0, 1}, 'Style', 'rotaryKnob'),...
            'InputChannels',1,...
            'OutputChannels',1,...
            'PluginName','SimpleDenoiser')
        
            %audioPluginParameter('MODELSELECT','DisplayName','Model','Mapping',{'enum','CNN','Feed Forward'}),...

    end

    properties (Constant)
        
        PluginConfig = audioPluginConfig( ...
            'DeepLearningConfig',coder.DeepLearningConfig('none'),...
            'CodeReplacementLibrary', '');

    end

    properties (Access = private)

        % stf Short-time Fourier transform (STFT) object. Converts the
        % audio time-series signal to STFT before feeding it to the
        % denoising neural network.
        stf
        % istf Inverse STFT (ISTFT) object. Converts the STFT at the output
        % of the neural network to a time-series audio signal.
        istf
        % segmentBuffer STFT segment buffer. Used to feed the 8 most recent
        % STFT segments to the neural network.
        segmentBuffer
        % bufferOut Buffer at the output of the neural network after ISTFT
        bufferOut
        % bufferNN Buffer at the output of src1
        bufferNN
        % buffer Buffer at output
        buffer
        % noise file
        audio
        idx

        % Sample-rate conversion utilities
        src16To8
        src441To8
        src48To8
        src96To8
        src192To8
        src32To8
        buffTo8

        src16From8
        src441From8
        src48From8
        src96From8
        src192From8
        src32From8
        buffFrom8

        % VADNet
        vadNet
        noiseGate

    end

    methods

        function plugin = SimpleDenoiser()
            % Initialize internal objects
            WindowLength = 256;
            win = hamming(WindowLength,'periodic');
            Overlap = round(0.75 * WindowLength);
            FFTLength = WindowLength;
            plugin.stf  = dsp.STFT(win,Overlap,FFTLength,...
                'FrequencyRange','onesided');
            plugin.istf = dsp.ISTFT(win,Overlap,...
                'WeightedOverlapAdd',false,...
                'FrequencyRange','onesided');

            plugin.segmentBuffer = dsp.AsyncBuffer('Capacity',8);
            plugin.bufferOut = dsp.AsyncBuffer;
            plugin.bufferNN = dsp.AsyncBuffer;
            plugin.buffer = dsp.AsyncBuffer;

            plugin.idx = 1;

            if strcmp(plugin.TrainingAudio, 'Washing Machine')
                plugin.audio = audioread("WashingMachine-16-8-mono-1000secs.mp3");
            elseif strcmp(plugin.TrainingAudio, 'White Noise')
                plugin.audio = audioread("WhiteNoise.wav");
            end

            plugin.src16From8 = dsp.FIRInterpolator(2);
            plugin.src441From8 = dsp.FIRRateConverter(441,80);
            plugin.src48From8 = dsp.FIRInterpolator(6);
            plugin.src96From8 = dsp.FIRInterpolator(12);
            plugin.src192From8 = dsp.FIRInterpolator(24);
            plugin.src32From8 = dsp.FIRInterpolator(4);
            plugin.buffFrom8 = dsp.AsyncBuffer;

            plugin.src16To8 = dsp.FIRDecimator(2);
            plugin.src441To8 = dsp.FIRRateConverter(80,441);
            plugin.src48To8 = dsp.FIRDecimator(6);
            plugin.src96To8 = dsp.FIRDecimator(12);
            plugin.src192To8 = dsp.FIRDecimator(24);
            plugin.src32To8 = dsp.FIRDecimator(4);
            plugin.buffTo8 = dsp.AsyncBuffer;

            plugin.vadNet = audioPretrainedNetwork('vadnet');
            plugin.noiseGate = noiseGate('SampleRate', getSampleRate(plugin));

        end

        function set.MODELSELECT(plugin,val)
            plugin.MODELSELECT = val;
        end

        function set.TrainingAudio(plugin, val)
            plugin.TrainingAudio = val;
        end

        function reset(plugin)

        end

        function out = process(plugin,in)
            if plugin.Bypass
                out = in;
            else
                if isempty(plugin.idx)
                    plugin.idx=1;
                end
                noise = plugin.audio(plugin.idx : plugin.idx + length(in) - 1);
                plugin.idx = plugin.idx + length(in);
                %in = in + noise * 2;
                in = in + noise * 0.1;
    
                % if strcmp(plugin.MODELSELECT, 'CNN')
                %     useCNN = 1;
                % else
                %     useCNN=0;
                % end
    
                dt = class(in);
                in = single(in);
    
                fs = getSampleRate(plugin);
    
                % Convert signal to 8 kHz
                x = convertTo8kHz(plugin, in, fs);

                % Write 8 kHz signal to buffer
                write(plugin.bufferNN,x(:,1:size(in,2)));
    
                % The denoising neural network operates on audio frames of
                % length 64.
                numSamples = double(plugin.bufferNN.NumUnreadSamples);
                numFrames = floor(numSamples/64);
    
                z = zeros(64*numFrames,1,'single');
    
                for index=1:numFrames
                    frame = read(plugin.bufferNN, 64);
                    sftSeg  = plugin.stf(frame);
                    sftSeg = sftSeg(1:129,1);
    
                    % Write most recent STFT vector to buffer
                    write(plugin.segmentBuffer,abs(sftSeg).');
    
                    % Read most recent 8 STFT vectors, with overlap of 7
                    SFFT_Image = read(plugin.segmentBuffer,8,7).';
    
                    % Denoise. Y is the STFT of the denoised frame
                    Y = denoise(reshape(SFFT_Image, [129 8 1]), useCNN).';
                    
                    ogMagnitude = abs(sftSeg);
                    Y = Y(:); % Make sure Y is a column vector
                    ogMagnitude = ogMagnitude(:); % Make sure ogMagnitude is a column vector
                    
                    blendFactor = plugin.Strength;
                    blendedMagnitude = (blendFactor * Y) + ((1 - blendFactor) * ogMagnitude);

                    % Inverse STFT. Use phase of noisy input audio
                    %isftSeg = Y.*exp(1j * angle(sftSeg));
                    isftSeg = blendedMagnitude.*exp(1j * angle(sftSeg));
                    z((index-1)*64+1:index*64) = plugin.istf(isftSeg);
                end          
    
                % Convert from 8 kHz back to the input sample rate
                y = convertFrom8kHz(plugin, z, fs);

                %VADNet processing
                vadFeatures = vadnetPreprocess(y, fs);
                probSpeech = predict(plugin.vadNet, vadFeatures);
                plugin.noiseGate.Threshold = -140 * mean(probSpeech);
                  
                % Process through noise gate
                y = plugin.noiseGate(y);
                     
                % Write to buffer
                write(plugin.buffer,y(:,1:size(in,2)));
    
                % Return output (same length as input)
                frameLength = size(in,1);
                out = cast(read(plugin.buffer,frameLength),dt);

            end
    
        end
    end

    methods (Access=protected)
        function y = convertTo8kHz(plugin,in, fs)
            % convertFrom8kHz Convert signal x from Fs to 8 kHz

            % Buffer input audio frame
            write(plugin.buffTo8, in);

            % The length of the input to the sample-rate converter must be
            % a multiple of the decimation factor
            frameLength = size(in,1);
            N = getSRCFrameLength(frameLength,fs);
            numSamples = double(plugin.buffTo8.NumUnreadSamples);
            L = floor(numSamples/N);
            if L>0
                toRead = L*N;
                x = read(plugin.buffTo8, toRead);
            else
                x = zeros(N,size(in,2),'like',in);
            end

            switch fs
                case {8000}
                    y = x;
                case {16000}
                    L = floor(length(x)/2);
                    z = x(1:L*2,:);
                    y = plugin.src16To8(z);
                case {44100}
                    % Keep the frame rate constant
                    L = size(x,1)/441;
                    y = zeros(L*80,size(x,2),'like',x);
                    for index=1:L
                        frame = x((index-1)*441+1:441*index,1:size(x,2));
                        y((index-1)*80+1:80*index,:) = plugin.src441To8(frame(1:441,:));
                    end
                case {48000}
                    L = floor(length(x)/6);
                    z = x(1:L*6,:);
                    y = plugin.src48To8(z);
                case {96000}
                    L = floor(length(x)/12);
                    z = x(1:L*12,:);
                    y = plugin.src96To8(z);
                case {192000}
                    L = floor(length(x)/24);
                    z = x(1:L*24,:);
                    y = plugin.src192To8(z);
                case {32000}
                    L = floor(length(x)/4);
                    z = x(1:L*4,:);
                    y = plugin.src32To8(z);
                otherwise
                    y = x;
            end
        end

        function y = convertFrom8kHz(plugin,in, fs)
            % convertFrom8kHz Convert signal x from 8 kHz to Fs

            write(plugin.buffFrom8,in);

            % The length of the input to the sample-rate converter must be
            % a multiple of the decimation factor
            numSamples = double(plugin.buffFrom8.NumUnreadSamples);
            if fs==44100
                toRead = floor(numSamples/80)*80;
            else
                toRead = numSamples;
            end
            x = read(plugin.buffFrom8,toRead);

            switch fs
                case {8000}
                    y = x;
                case {16000}
                    y = plugin.src16From8(x);
                case {44100}
                    % Keep the frame length constant
                    L = size(x,1)/80;
                    y = zeros(L*441,size(x,2),'like',x);
                    for index=1:L
                        frame = x((index-1)*80+1:80*index,:);
                        y((index-1)*441+1:441*index,:) = plugin.src441From8(frame(1:80,:));
                    end
                case {48000}
                    y = plugin.src48From8(x);
                case {96000}
                    y = plugin.src96From8(x);
                case {192000}
                    y = plugin.src192From8(x);
                case {32000}
                    y = plugin.src32From8(x);
                otherwise
                    y = x;
            end

        end
    end

end

%function y = denoise(x, useCNN)
function y = denoise(x)
% Denoise input frame
    % Load the pre-trained network. 
    persistent noisyMean noisyStd cleanMean cleanStd;
    if isempty(cleanMean)
        s = load("denoisenet.mat");
        cleanMean = s.cleanMean;
        cleanStd = s.cleanStd;
        noisyMean = s.noisyMean;
        noisyStd = s.noisyStd;
    end
    
    x = (x-noisyMean)/noisyStd;
    
    persistent trainedNetCNN trainedNetFF
    if isempty(trainedNetFF)
        trainedNetFF = s.denoiseNetFullyConnected;
        trainedNetCNN = s.denoiseNetFullyConvolutional;
    end
    
    %if useCNN
        y = predict(trainedNetCNN,x)';
    % else
    %     y = predict(trainedNetFF,x);
    % end

    y = y*cleanStd+cleanMean;

end

function N = getSRCFrameLength(L,fs)
    switch fs
        case {8000}
            N = L;
        case {16000}
            N = 2;
        case {44100}
            N = 441;
        case {48000}
            N = 6;
        case {96000}
            N = 12;
        case {192000}
            N = 24;
        case {32000}
            N = 4;
        otherwise
            N = L;
    end
end