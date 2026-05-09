def log_metrics(writer, metrics, step):
    for k, v in metrics.items():
        writer.add_scalar(k, v, step)


def log_learning_curves(writer, train_loss, val_loss, step):
    writer.add_scalar('Loss/train', train_loss, step)
    writer.add_scalar('Loss/val', val_loss, step)
