function cnnnumgradcheck(net, x, y)
    epsilon = 1e-4;
    er      = 1e-8;
    n = numel(net.layers);

    % Check gradients for biases in the fully connected layer
    for j = 1 : numel(net.ffb)
        % Create two network copies: net_m and net_p
        net_m = net; 
        net_p = net;
        % Add epsilon to net_p biases and subtract epsilon from net_m biases
        net_p.ffb(j) = net_m.ffb(j) + epsilon;
        net_m.ffb(j) = net_m.ffb(j) - epsilon;
        % Forward and backward propagation
        net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
        net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
        % Compute numerical gradient
        d = (net_p.L - net_m.L) / (2 * epsilon);
        % Compute the error between numerical gradient and backprop gradient
        e = abs(d - net.dffb(j));
        if e > er
            error('Numerical gradient checking failed');
        end
    end

    % Check gradients for weights in the fully connected layer
    for i = 1 : size(net.ffW, 1)
        for u = 1 : size(net.ffW, 2)
            % Create two network copies: net_m and net_p
            net_m = net; 
            net_p = net;
            % Add and subtract epsilon to/from the weights
            net_p.ffW(i, u) = net_m.ffW(i, u) + epsilon;
            net_m.ffW(i, u) = net_m.ffW(i, u) - epsilon;
            % Forward and backward propagation
            net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
            net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
            % Compute the error between numerical gradient and backprop gradient
            d = (net_p.L - net_m.L) / (2 * epsilon);
            e = abs(d - net.dffW(i, u));
            if e > er
                error('Numerical gradient checking failed');
            end
        end
    end

    % Check gradients for each layer in the network
    for l = n : -1 : 2
        if strcmp(net.layers{l}.type, 'c') % Convolutional layer
            for j = 1 : numel(net.layers{l}.a)
                % Create two network copies: net_m and net_p
                net_m = net; net_p = net;
                % Add and subtract epsilon to/from the biases
                net_p.layers{l}.b{j} = net_m.layers{l}.b{j} + epsilon;
                net_m.layers{l}.b{j} = net_m.layers{l}.b{j} - epsilon;
                % Forward and backward propagation
                net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
                net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
                % Compute the error between numerical gradient and backprop gradient
                d = (net_p.L - net_m.L) / (2 * epsilon);
                e = abs(d - net.layers{l}.db{j});
                if e > er
                    error('Numerical gradient checking failed');
                end

                % Check gradients for kernels in the convolutional layer
                for i = 1 : numel(net.layers{l - 1}.a)
                    % Iterate over rows and columns of the kernel
                    for u = 1 : size(net.layers{l}.k{i}{j}, 1)
                        for v = 1 : size(net.layers{l}.k{i}{j}, 2)
                            net_m = net; 
                            net_p = net;
                            % Add and subtract epsilon to/from the kernel weights
                            net_p.layers{l}.k{i}{j}(u, v) = net_p.layers{l}.k{i}{j}(u, v) + epsilon;
                            net_m.layers{l}.k{i}{j}(u, v) = net_m.layers{l}.k{i}{j}(u, v) - epsilon;
                            % Forward and backward propagation
                            net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
                            net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
                            % Compute the error between numerical gradient and backprop gradient
                            d = (net_p.L - net_m.L) / (2 * epsilon);
                            e = abs(d - net.layers{l}.dk{i}{j}(u, v));
                            if e > er
                                error('Numerical gradient checking failed');
                            end
                        end
                    end
                end
            end
        elseif strcmp(net.layers{l}.type, 's') % Subsampling layer (if applicable)
%            for j = 1 : numel(net.layers{l}.a)
%                net_m = net; net_p = net;
%                net_p.layers{l}.b{j} = net_m.layers{l}.b{j} + epsilon;
%                net_m.layers{l}.b{j} = net_m.layers{l}.b{j} - epsilon;
%                net_m = cnnff(net_m, x); net_m = cnnbp(net_m, y);
%                net_p = cnnff(net_p, x); net_p = cnnbp(net_p, y);
%                d = (net_p.L - net_m.L) / (2 * epsilon);
%                e = abs(d - net.layers{l}.db{j});
%                if e > er
%                    error('Numerical gradient checking failed');
%                end
%            end
        end
    end
%    keyboard
end
