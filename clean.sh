# Remove temporary test files
rm *.jpg

# Remove all of the atlas SIFT files
find dataset/atlas_swanson -name "*.sift" -delete
find dataset/atlas_pw -name "*.sift" -delete

# Done!
read -t5 -n1 -r -p 'Done cleaning...' key