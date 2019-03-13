function w = nnhedge_weights(r, scale, A)
n = numel(r);
w = zeros(1,n);

for i = 1:n
    if (r(i) + A <= 0)
        w(i) = 0;
    else
        w(i) =  (r(i) + A)/scale*exp( (r(i) + A)* (r(i) + A) / (2 * scale));
    end
end

