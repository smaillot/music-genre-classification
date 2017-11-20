function [] = plot_PC_againt_all(genre_features, features, PC_mat)
    hold on
    data_features = features * PC_mat;
    data_genre = genre_features * PC_mat;
    plot(data_features(:,1), data_features(:,2), '.r', 'MarkerSize', 32)
    plot(data_genre(:,1), data_genre(:,2), '.b', 'MarkerSize', 32)
end