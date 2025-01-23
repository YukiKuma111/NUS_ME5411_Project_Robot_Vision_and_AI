function X=flipall(X)
    % Flip the matrix across all dimensions
    for i=1:ndims(X)
        X = flip(X,i);
    end
end