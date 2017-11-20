%% load features

if exist('features') == 0
    load('features.mat');
    if size(features, 1) == 83
        features=[features(1:31,:);features(33:35,:);features(37:end,:)];
    end
end

if exist('PC_mat') == 0
    [coef, score, latent] = pca(features);
    
    plot(cumsum(latent/sum(latent)))
    hold on
    plot([0 size(features,1)],[0.95 0.95])
    plot([0 size(features,1)],[0.9 0.9])
    plot([0 size(features,1)],[0.8 0.8])

    reduction = 2;

    PC_mat = coef(:,1:reduction);
end

%% plot all genres

f = figure
hold on
set(f, 'Position', [0, 400, 2000, 800]);

add_genre_to_plot(classical, PC_mat)
add_genre_to_plot(country, PC_mat)
add_genre_to_plot(electro, PC_mat)
add_genre_to_plot(folk, PC_mat)
add_genre_to_plot(hiphop, PC_mat)
add_genre_to_plot(indie, PC_mat)
add_genre_to_plot(jazz, PC_mat)
add_genre_to_plot(lounge, PC_mat)
add_genre_to_plot(pop, PC_mat)
add_genre_to_plot(rock, PC_mat)
title('Genre plot in PC2 space')

legend('classical', 'country', 'electro', 'folk', 'hiphop', 'indie', 'jazz', 'lounge', 'pop', 'rock')

%% plot againt all

f = figure
hold on
set(f, 'Position', [0, 0, 2000, 800]);

subplot(2, 5, 1)
plot_PC_againt_all(classical, features, PC_mat)
title('classical')

subplot(2, 5, 2)
plot_PC_againt_all(country, features, PC_mat)
title('country')

subplot(2, 5, 3)
plot_PC_againt_all(electro, features, PC_mat)
title('electro')

subplot(2, 5, 4)
plot_PC_againt_all(folk, features, PC_mat)
title('folk')

subplot(2, 5, 5)
plot_PC_againt_all(hiphop, features, PC_mat)
title('hiphop')

subplot(2, 5, 6)
plot_PC_againt_all(indie, features, PC_mat)
title('indie')

subplot(2, 5, 7)
plot_PC_againt_all(jazz, features, PC_mat)
title('jazz')

subplot(2, 5, 8)
plot_PC_againt_all(lounge, features, PC_mat)
title('lounge')

subplot(2, 5, 9)
plot_PC_againt_all(pop, features, PC_mat)
title('pop')

subplot(2, 5, 10)
plot_PC_againt_all(rock, features, PC_mat)
title('rock')