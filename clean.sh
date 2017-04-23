# Remove temporary test files
rm *.jpg

# Remove all of Swanson's SIFT files
find atlas_swanson -name "*.sift" -delete

# Done!
read -t5 -n1 -r -p 'Done cleaning...' key