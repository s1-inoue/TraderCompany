kaggle datasets download -d andrewmvd/sp-500-stocks -p ./input
unzip ./input/sp-500-stocks.zip -d ./input
rm ./input/sp-500-stocks.zip

source ./src/utils/envinit.py