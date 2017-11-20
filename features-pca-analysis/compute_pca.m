%% delete invalids - execute once

%features=[features(1:31,:);features(33:35,:);features(37:end,:)];

%% Reduced

reduced = [features(:,1:31) features(:,1201)];

%% rythm

A = features(:,32:1055);

[coef,score,latent] = pca(A);
% Check: Projection of data in the PCA basis is given by (A-mean(A))*evec:
proj =  (A-repmat(mean(A),size(A,1),1))*coef;

% Check: Eigenvalues and eigenvectos come from diagonalization of
% covariance matrix:
%S = cov(A);
%[V,D] = eig(S);

plot(cumsum(latent/sum(latent)))
hold on
plot([0 size(A,1)],[0.95 0.95])
plot([0 size(A,1)],[0.9 0.9])
plot([0 size(A,1)],[0.8 0.8])

reduction = uint8(str2double(input('Number of features to keep : ','s')));

B = A * coef(:,1:reduction);
%figure()
%plotmatrix(B)
reduced = [reduced B];

%% melody

A = features(:,1056:1200);

[coef,score,latent] = pca(A);

plot(cumsum(latent/sum(latent)))
hold on
plot([0 size(A,1)],[0.95 0.95])
plot([0 size(A,1)],[0.9 0.9])
plot([0 size(A,1)],[0.8 0.8])

reduction = uint8(str2double(input('Number of features to keep : ','s')));

B = A * coef(:,1:reduction);
%figure()
%plotmatrix(B)
reduced = [reduced B];

%% Final PCA

[coef,score,latent] = pca(reduced);

plot(cumsum(latent/sum(latent)))
hold on
plot([0 size(A,1)],[0.95 0.95])
plot([0 size(A,1)],[0.9 0.9])
plot([0 size(A,1)],[0.8 0.8])

reduction = uint8(str2double(input('Number of features to keep : ','s')));

final = reduced * coef(:,1:reduction);

figure()
plotmatrix(final)

%% All data PCA

[coef, score, latent] = pca(features);
    
plot(cumsum(latent/sum(latent)))
hold on
plot([0 size(features,1)],[0.95 0.95])
plot([0 size(features,1)],[0.9 0.9])
plot([0 size(features,1)],[0.8 0.8])

reduction = uint8(str2double(input('Number of features to keep : ','s')));

direct = coef(:,1:reduction);

%%figure()
%%plotmatrix(features * direct)

%% test


pca_electro = electro * direct;
pca_rock = rock * direct;
pca_jazz = jazz * direct;
pca_classical = classical * direct;

n1 = 10;
n2 = 10;
os1 = 0;
os2 = 0;
PC1 = 1+os1:n1+os1;
PC2 = 1+os2:n2+os2;

figure

for pc1=PC1
    for pc2=PC2
        subplot(n1,n2,(pc1-os1)+((pc2-os2)-1)*n1)
        hold on
        plot(pca_electro(:,pc1), pca_electro(:,pc2),'.r','MarkerSize',15)
        plot(pca_rock(:,pc1), pca_rock(:,pc2),'.g','MarkerSize',15)
        plot(pca_jazz(:,pc1), pca_jazz(:,pc2), '.b','MarkerSize',15)
        plot(pca_classical(:,pc1), pca_classical(:,pc2), '.k','MarkerSize',15)
    end
end

%% in 3D

pc = [1, 3, 4];

plot3(pca_electro(:,pc(1)),pca_electro(:,pc(2)),pca_electro(:,pc(3)),'.r')
hold on;
plot3(pca_rock(:,pc(1)),pca_rock(:,pc(2)),pca_rock(:,pc(3)),'.g')
plot3(pca_jazz(:,pc(1)),pca_jazz(:,pc(2)),pca_jazz(:,pc(3)),'.b')
plot3(pca_classical(:,pc(1)),pca_classical(:,pc(2)),pca_classical(:,pc(3)),'xk')
axis equal