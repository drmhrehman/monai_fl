import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from pathlib import Path
from monai.inferers import sliding_window_inference
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def main(config):
    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # overload with my own data-dirs
    #ims = ['/home/mark/projects/wellcome/ich_segmentation/data/yee/processed/wroropRJZ2958331.nii']
    #segs = ['/home/mark/projects/wellcome/ich_segmentation/data/yee/processed/wropRJZ2958331_haemorrhage.nii']

    ims = list(Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/processed/images/').glob('wcrroropRJ*.nii'))
    #ims = list(Path('/home/mark/projects/wellcome/ich_segmentation/data/yee/processed_thincut/').rglob('wroropRJ*.nii'))
    ims = [str(i) for i in ims]
    segs = [i.replace('/images/','/labels/') for i in ims]
    image_and_labels = []
    for i, s in zip(ims, segs):
        image_and_labels.append({'img':i,
                                 'seg':s})
    from monai.data import Dataset
    dataset= Dataset(image_and_labels, transform=data_loader.val_transforms)

    from base import BaseDataLoader
    test_data_loader = BaseDataLoader(dataset,batch_size=1, shuffle=False, validation_split=0.0, num_workers=1)
    # build model architecture
    model = config.init_obj('arch', module_arch)
    #logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    # if any of the criterion/metrics are classes (rather than functions), iniitalise
    if isinstance(loss_fn, type):
        loss_fn = loss_fn()
    for m in metric_fns:
        if isinstance(m, type):
            m = m()

    # set up logging
    config.make_experiment_dirs()
    logger = config.get_logger('test')

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # set up directorys for saving results
    result_stem = Path(config['trainer']['save_dir']) / 'results'
    config_path = Path(config.resume)
    results_dir = result_stem / config_path.parent.parent.name / config_path.parent.name
    results_dir.mkdir(parents=True, exist_ok=True)

    dice_metric = {}
    volume_vs_dice = []
    subject_vs_dice = []

    with torch.no_grad():
        for i, (data_batch) in enumerate(tqdm(test_data_loader)):
            data, target = data_batch['img'].to(device), data_batch['seg'].to(device)
            batch_size = data.shape[0]

            output_logits = sliding_window_inference(data,
                                                     sw_batch_size=4,
                                                     roi_size=(128, 128, 128),
                                                     predictor=model,
                                                     overlap=0.25,
                                                     do_sigmoid=False)
            output = torch.sigmoid(output_logits)

            hard_label = (output > 0.5).type(torch.int8)
            # save sample images, or do something with output here
            image_filenames = data_batch['img_meta_dict']['filename_or_obj']
            for item in range(batch_size):
                subject_id = Path(image_filenames[item]).stem.split('.')[0]
                nifti_img = nib.Nifti1Image(data[item, 0,  ...].cpu().numpy().squeeze(), np.eye(4))
                nifti_img.to_filename(str(results_dir / f'{subject_id}.nii.gz'))
                # nifti_label = nib.Nifti1Image(output[item, ...].cpu().numpy().squeeze(), np.eye(4))
                # nifti_label.to_filename(str(results_dir / f'{subject_id}_label.nii.gz'))
                nifti_hard_label = nib.Nifti1Image(hard_label[item, ...].cpu().numpy().squeeze(), np.eye(4))
                nifti_hard_label.to_filename(str(results_dir / f'{subject_id}_hard_label.nii.gz'))

                nifti_gt = nib.Nifti1Image(target[item, ...].cpu().numpy().squeeze(), np.eye(4))
                nifti_gt.to_filename(str(results_dir / f'{subject_id}_gt.nii.gz'))
            # computing loss, metrics on test set
            loss = loss_fn(output, target)

            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target).cpu() * batch_size

            # get dice for each sample
            from monai.metrics import compute_meandice
            dice = compute_meandice(output, target, include_background=False)
            for item in range(batch_size):
                subject_id = Path(image_filenames[item]).stem.split('.')[0]
                dice_metric[subject_id] = dice[item].item()
                volume_vs_dice.append([target[item, ...].sum().cpu().numpy(), dice[item].item()])
                subject_vs_dice.append([Path(image_filenames[i]).stem, dice[item].item()])

    # make plot of dice overlap per subject
    fig, ax = plt.subplots(figsize=(8,5))
    data = np.array(volume_vs_dice)
    ax.scatter(data[:,0], data[:,1], marker='+', s=90, linewidths=3)
    ax.set_xlabel('Ground truth haemorrhage size / voxels')
    ax.set_ylabel('Dice overlap')
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.1))
    ax.set_ylim([0,1])
    ax.yaxis.grid()
    plt.savefig(results_dir /'volume_vs_dice_test.png')

    fig, ax = plt.subplots(figsize=(8,10))
    data = np.array(subject_vs_dice)
    ax.bar(data[:,0], data[:,1].astype(float))
    plt.xticks(rotation=90)
    ax.yaxis.set_ticks(np.arange(0, 1.01, 0.1))
    ax.set_ylim([0,1])
    ax.yaxis.grid()
    plt.tight_layout()
    plt.savefig(results_dir /'subject_vs_dice_test.png')


    for key, value in dice_metric.items():
        print(f'{key}: {value:.2f}')
    print(f'Mean dice: {np.mean(list(dice_metric.values()))}')

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
