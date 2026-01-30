echo "In order to run Imitator, you need to download FLAME. Before you continue, you must register and agree to license terms at:"
echo -e '\e]8;;https://flame.is.tue.mpg.de\ahttps://flame.is.tue.mpg.de\e]8;;\a'

while true; do
    read -p "I have registered and agreed to the license terms at https://flame.is.tue.mpg.de? (y/n)" yn
    case $yn in
        [Yy]* ) break;;
        [Nn]* ) exit;;
        * ) echo "Please answer yes or no.";;
    esac
done


flame_path=FLAMEModel
echo "Downloading FLAME related assets"
rm -rf $flame_path
mkdir -p $flame_path
wget "https://keeper.mpdl.mpg.de/f/02e901aa30d24f509a5a/?dl=1" -O FLAME.zip
echo "Extracting FLAME..."
unzip FLAME.zip -d $flame_path
rm FLAME.zip

echo "Downloading 3DiFACE pretrained model..."
wget "https://keeper.mpdl.mpg.de/f/508fcbc9fee14f6aac1b/?dl=1" -O pretrained_models.zip
echo "Extracting pretrained_model..."
unzip pretrained_models.zip
rm pretrained_models.zip 
rm -rf __MACOSX
