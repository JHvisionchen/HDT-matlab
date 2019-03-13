function [positions] = tracker_ensemble(img_files, pos, target_sz, ...
                                                padding, lambda, output_sigma_factor, interp_factor, ...
                                                cell_size, show_visualization, rect_anno, bSaveImage, pathModel)


initial_net(pathModel);
nlayers = [37, 34, 32, 28, 25, 23];
A = 0.011; % relaxed factor
im_sz = size(imread(img_files{1}));
window_sz = get_search_window(target_sz,im_sz, padding);
output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;

% Create regression labels, gaussian shaped, with a bandwidth
% proportional to target size
l1_patch_num = floor(window_sz / cell_size);
yf = fft2(gaussian_shaped_labels(output_sigma, l1_patch_num));

% Store pre-computed cosine window
cos_window = hann(size(yf,1)) * hann(size(yf, 2))';

if show_visualization
    update_visualization = show_video(img_files,'');
end


% Variables ending with 'f' are in the Fourier domain.

positions  = zeros(numel(img_files), 2);  

alphaf = cell(1,length(nlayers));
model_xf = cell(1,length(nlayers));
model_alphaf = cell(1,length(nlayers));

for frame = 1:numel(img_files),
    
    im = imread(img_files{frame});
    if ismatrix(im),
        im = cat(3, im, im, im);
    end
    
    if frame >1
        % Obtain a subwindow for detection at the position from last
        % frame, and convert to Fourier domain (its size is unchanged)
        patch = get_subwindow(im, pos, window_sz);
        feat   = get_features(patch, cos_window, nlayers);
        
        for ii = 1:length(nlayers)
            
            zf   = fft2(feat{ii});
            kzf = sum(zf .* conj(model_xf{ii}), 3) / numel(zf);
            
            response{ii} = real(fftshift(ifft2(model_alphaf{ii} .* kzf)));  % weak trackers
            maxres(ii)   = max(response{ii}(:));
            [row,col]        = find(response{ii}==maxres(ii),1);
            experts{ii}.row = row;
            experts{ii}.col = col;
        end
        
        if frame == 2 % to adjust w
            rect = rect_anno(2,:); % The used gt at the 2nd frame can be replaced by gt in the 1st frame since 
                                                 % most of the targets in videos move slightly  between frames. 
                                                 % It can also be obtained  by another tracker user specified.

            pos2(1) = rect(2) + floor(rect(4)/2);
            pos2(2) = rect(1) + floor(rect(3)/2);
            row = (pos2(1) - pos(1,1))/cell_size + 1 + floor(size(zf,1)/2);
            col = (pos2(2) - pos(1,2))/cell_size + 1 + floor(size(zf,2)/2);
        else
            row = 0; col = 0;
            for ii = 1:length(nlayers)
                row = row + W(ii)*experts{ii}.row;
                col = col + W(ii)*experts{ii}.col;
            end
        end
        
        vert_delta = row; horiz_delta = col;
        vert_delta = vert_delta - floor(size(zf,1)/2);
        horiz_delta = horiz_delta - floor(size(zf,2)/2);
        
        pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
        
    end
    
    % Obtain a subwindow for training at newly estimated target position
    
    patch = get_subwindow(im, pos, window_sz);
    feat = get_features(patch, cos_window, nlayers);
    
    % Fast training with new observations
    for ii = 1:length(nlayers)
        xf{ii} = fft2(feat{ii});
        kf = sum(xf{ii} .* conj(xf{ii}), 3) / numel(xf{ii});
        alphaf{ii} = yf./ (kf + lambda);
    end
    
    
    if frame == 1,  % First frame, train with a single image
        
        for ii=1:length(nlayers)
            model_alphaf{ii} = alphaf{ii};
            model_xf{ii} = xf{ii};
        end
        
        W = [1 0.2 0.2 0.02 0.03 0.01];
        R(1:length(nlayers)) = 0;
        loss(1:6,1:length(nlayers)) = 0;
        
    else % Update trackers
        
        for ii = 1:length(nlayers)
            model_alphaf{ii} = (1 - interp_factor) * model_alphaf{ii} + interp_factor * alphaf{ii};
            model_xf{ii} = (1 - interp_factor) * model_xf{ii} + interp_factor * xf{ii};
        end
        
    end
    
    
    % Save position and timing
    positions(frame,:) = pos;
    
    % Visualization
    if show_visualization
        
        box = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        
        stop = update_visualization(frame, box);
        if stop, break, end  % User pressed Esc, stop early
        
        drawnow
        if bSaveImage
            imwrite(frame2im(getframe(gcf)), ['./result/' num2str(frame) '.jpg']);
            pause(0.05)  %uncomment to run slower
        end
    end
    %%%%%%%%%%%%% update weights with hedging %%%%%%%%%%%%%%%%%%
    if frame > 1
        % Compute loss of each weak tracker
        row = round(row); col = round(col);
        
        for ii = 1:length(nlayers)
            loss(6,ii) = maxres(ii) - response{ii}(row,col);
        end
        lossA = sum(W.*loss(6,:));
        LosIdx = mod(frame-1,5)+1;
        
        LosMean = mean(loss(1:5,:));
        LosStd = std(loss(1:5,:));
        LosMean(LosMean<0.0001) = 0;
        LosStd(LosStd<0.0001) = 0;
        
        curDiff = loss(6,:)-LosMean;
        alpha=0.97*exp((-10*abs(curDiff)./(LosStd+eps)));
        
        % Truncation
        alpha(alpha>0.9)=0.97;
        alpha(alpha<0.12)=0.119;
        
        R=R.*(alpha)+(1-alpha).*(lossA-loss(6,:));
        
        % Update loss history
        loss(LosIdx,:)=loss(6,:);
        
        c = find_nh_scale(R, A);
        W = nnhedge_weights(R, c, A);
        W = W / sum(W);
    end
    
end

close all

end
