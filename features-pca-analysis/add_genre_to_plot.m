function [] = add_genre_to_plot(genre_features, PC_mat)
    data = genre_features * PC_mat;
    plot(data(:,1), data(:,2), '.', 'MarkerSize', 32)
end