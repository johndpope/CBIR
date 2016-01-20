% Courtesy of Color-Based Segmentation Using K-Means Clustering
% http://in.mathworks.com/help/images/examples/color-based-segmentation-using-k-means-clustering.html
function kmeansSegmentation(kclusters, filename)
delete(findall(0,'Type','figure'))
he = imread(filename);
imshow(he), title('Shekhawati Painting');
text(size(he,2),size(he,1)+15,...
     'Image courtesy of Google Images', ...
     'FontSize',7,'HorizontalAlignment','right');
cform = makecform('srgb2lab');
lab_he = applycform(he,cform);
ab = double(lab_he(:,:,2:3));
nrows = size(ab,1);
ncols = size(ab,2);
ab = reshape(ab,nrows*ncols,2);

nColors = kclusters;
% repeat the clustering 3 times to avoid local minima
[cluster_idx, cluster_center] = kmeans(ab,nColors,'distance','sqEuclidean', ...
                                      'Replicates',3);
pixel_labels = reshape(cluster_idx,nrows,ncols);
figure;
imshow(pixel_labels,[]), title('image labeled by cluster index');
segmented_images = cell(1,3);
rgb_label = repmat(pixel_labels,[1 1 3]);

for k = 1:nColors
    color = he;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color;
end

callstr = 'set(gcbf,''Userdata'',double(get(gcbf,''Currentcharacter''))) ; uiresume ' ;
fh = figure('name','Segmented Clusters', ...
    'KeyPressFcn',@(obj,evt) 0);
imshow(segmented_images{1}), title('objects in cluster 1');
cnt = 0;
while(true)
    waitfor(gcf,'CurrentCharacter');
    key = uint8(get(gcf,'CurrentCharacter'));
    set(gcf,'CurrentCharacter', 'a');
    if((key == 'n') || (key == 'N'))
        cnt = cnt + 1;
    end
    if((key == 'p') || (key == 'P'))
        cnt = cnt - 1;
    end
    if(key == 'Q' || (key == 'q'))
        delete(findall(0,'Type','figure'))
        return
    end
    cnt = mod(cnt, nColors);
    imshow(segmented_images{cnt + 1}), title(sprintf('objects in cluster %d', cnt + 1));
end
end