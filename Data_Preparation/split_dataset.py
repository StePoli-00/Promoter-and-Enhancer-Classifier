import os
import shutil
import zipfile
import argparse
#nome del file zip da unzippare (se c'è)
zip_path=""
#nome del dataset che vuoi creare
dataset_name=""

#nome della cartella dell'embedding <nome embeddings>->embeddings->labels
embedding_name=""

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Parser for create Dataset')
    parser.add_argument('-e',"--emb", help='Folder con gli embeddings e label')
    parser.add_argument('-d', '--dataset', help='Dataset name')
    parser.add_argument('-z',"--zip",help="zip file, se si passa lo zip -i deve avere lo stesso nome di -z senza .zip ")
    
    args=parser.parse_args()
    embedding_name=args.emb
    dataset_name=args.dataset
    zip_path=args.zip
    training_path=os.path.join(dataset_name,"training_data")
    validation_path=os.path.join(dataset_name,"validation_data")
    testing_path=os.path.join(dataset_name,"testing_data")
    
    if embedding_name is None or  dataset_name is None:
        parser.print_help()
        parser.error("Parameters error")
        
    

    if not os.path.exists(os.path.join(".",dataset_name)):
        if zip_path!=None:
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    
                    zip_ref.extractall(".")
                print("File zip decompresso con successo.")
            except zipfile.BadZipFile:
                print("Il file specificato non è un file zip valido.")
        os.rename(embedding_name,"training_data")
        os.makedirs(dataset_name,exist_ok=True)
        shutil.move("training_data", os.path.join(dataset_name,))
    
        os.makedirs(os.path.join(testing_path),exist_ok=True)
        os.makedirs(os.path.join(testing_path,"embeddings"), exist_ok=True)
        os.makedirs(os.path.join(testing_path,"labels"),exist_ok=True)
        os.makedirs(os.path.join(validation_path),exist_ok=True)
        os.makedirs(os.path.join(validation_path,"embeddings"),exist_ok=True)
        os.makedirs(os.path.join(validation_path,"labels"),exist_ok=True)


    training_embeddings_path = os.path.join(training_path, "embeddings")
    training_labels_path = os.path.join(training_path, "labels")

    validation_embeddings_path = os.path.join(validation_path, "embeddings")
    validation_labels_path = os.path.join(validation_path, "labels")

    testing_embeddings_path = os.path.join(testing_path, "embeddings")
    testing_labels_path = os.path.join(testing_path, "labels")



    if (len(os.listdir(testing_embeddings_path))==0 and len(os.listdir(validation_embeddings_path))==0):

        for i, el in enumerate(os.listdir(training_embeddings_path)):
            if i % 10 == 0:
                shutil.move(os.path.join(training_embeddings_path, el), validation_embeddings_path)
                shutil.move(os.path.join(training_labels_path, el), validation_labels_path)
            if i % 10 == 1:
                shutil.move(os.path.join(training_embeddings_path, el), testing_embeddings_path)
                shutil.move(os.path.join(training_labels_path, el), testing_labels_path)


    list_path=[training_path,testing_path,validation_path] 
    for folder in list_path:
        num=len(os.listdir(os.path.join(folder,"embeddings")))
        print(f"num of data in {folder} folder: {num}")