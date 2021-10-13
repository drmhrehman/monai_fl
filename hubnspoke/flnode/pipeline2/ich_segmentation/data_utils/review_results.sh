path=/Users/mark/Downloads/BasicUnet_rigid_processing/0625_143241
ID=$1
mrview $path/${ID}.nii.gz  \
-interpolation 0 \
\
-overlay.load  $path/${ID}_gt.nii.gz \
-overlay.colour 0,200,0 \
-overlay.interpolation 0 \
\
-overlay.load  $path/${ID}_hard_label.nii.gz \
-overlay.colour 0,0,200 \
-overlay.interpolation 0 