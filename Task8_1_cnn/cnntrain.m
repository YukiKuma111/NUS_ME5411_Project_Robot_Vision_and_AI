function net = cnntrain(net, x, y, opts)
    % Get the number of samples (m)
    m = size(x, 3);
    
    % Calculate the number of batches
    numbatches = m / opts.batchsize;
    
    % Check if the number of batches is an integer
    if rem(numbatches, 1) ~= 0
        error('numbatches not integer');
    end
    
    % Initialize the loss history
    net.rL = [];
    
    for i = 1 : opts.numepochs
        % Display the current epoch number
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)]);
        
        % Start the timer for the current epoch
        tic;
        
        % Shuffle the sample indices randomly
        kk = randperm(m);
        
        for l = 1 : numbatches
            % Get the current batch of input and output data
            batch_x = x(:, :, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            batch_y = y(:, kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize));
            
            % Perform forward propagation
            net = cnnff(net, batch_x);
            
            % Perform backward propagation to compute gradients
            net = cnnbp(net, batch_y);
            
            % Apply the gradients to update the network parameters (weights and biases)
            net = cnnapplygrads(net, opts);
            
            % Record the initial loss (for the first epoch)
            if isempty(net.rL)
                net.rL(1) = net.L;
            end
            
            % Record the current loss (using exponential smoothing)
            net.rL(end + 1) = 0.99 * net.rL(end) + 0.01 * net.L;
        end
        
        % Stop the timer for the current epoch
        toc;
    end
end
