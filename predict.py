"""import libraries for network prediction purpose."""
import torch 
import argparse
from PIL import Image
import torchvision
import json
from torchvision import datasets, transforms, models
 
def define_command_parser():
    parser = argparse.ArgumentParser(description='predict your data with trained model')

    parser.add_argument('img_directory', action='store')
    parser.add_argument('checkpoint_directory', action='store')
    parser.add_argument('--top_k', action='store', dest='top_k', type=int, default=5)
    parser.add_argument('--gpu', action='store_true', dest='gpu', default=False)
    parser.add_argument('--category_names', action='store', dest='category_names', default=None)
    
    return parser

def load_model(checkpoint_path, device):
    checkpoint_ld = torch.load(checkpoint_path, map_location=device)
    if(checkpoint_ld['class_name'] == 'vgg16'):
        model = models.vgg16(pretrained=True)
    elif(checkpoint_ld['class_name'] == 'vgg13'):
        model = models.vgg13(pretrained=True)
    else:
        raise ValueError('Unspected network architecture', checkpoint_ld['class_name'])
    
    model.classifier = checkpoint_ld['classifier']
    model.class_to_idx = checkpoint_ld['class_to_idx']
    model.load_state_dict(checkpoint_ld['state_dict'])

    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image_path)
    img_loader = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    [0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225]
                                )
                            ])
    
    img_normailzed = img_loader(img)
    img_np = img_normailzed.numpy()
  
    return img_np

def predict(image_path, model, device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    # 1.Forward prediction
    model.to(device=device)
    img = process_image(image_path)
    img_tensor = torch.tensor(img)
    img_tensor.unsqueeze_(0)
    img_tensor = img_tensor.to(device=device)
    with torch.no_grad():
        output = torch.exp(model.forward(img_tensor))
    
    # 2.Top k list
    topk_prob, topk_classes = torch.topk(output, topk) 
    
    return topk_prob, topk_classes

def mapping(topk_classes, cat_to_name_path):
    with open(cat_to_name_path, 'r') as f:
        cat_to_name = json.load(f)
    labels = [cat_to_name[str(x)] for x in topk_classes]
    return labels

def main():
    print('Pytorch version: ', torch.__version__)
    print('Torchvision version: ', torchvision.__version__)

    # 1.Get the arguments and decide which devices to be used
    args = define_command_parser().parse_args()
    args.device = None
    if args.gpu and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    
    # 2.Load model
    model = load_model(args.checkpoint_directory, args.device)

    # 3.Predict the image
    topk_prob, topk_classes = predict(args.img_directory, model, device=args.device, topk=args.top_k)
    prob_cpu = topk_prob.data.cpu().numpy()[0]
    clas_cpu = topk_classes.data.cpu().numpy()[0]
    if args.category_names:
        # 4.mapping from index to classes
        labels = mapping(clas_cpu, args.category_names)

    # 4.Print the result
    print("The top{} probability predicted are: ".format(args.top_k), prob_cpu) 
    print("The indexes are: ", clas_cpu)
    if args.category_names:
        print("Categories are: ", labels)
    
if __name__ == '__main__':
    main()