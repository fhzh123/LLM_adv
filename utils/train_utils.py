def input_to_device(batch_iter, device):

    if len(batch_iter[0]) == 3:

        src_sequence = batch_iter[0][0]
        src_att = batch_iter[0][1]
        src_token_type = batch_iter[0][2]
        trg_label = batch_iter[1]

        src_sequence = src_sequence.to(device, non_blocking=True)
        src_att = src_att.to(device, non_blocking=True)
        src_token_type = src_token_type.to(device, non_blocking=True)
        trg_label = trg_label.to(device, non_blocking=True)

    return (src_sequence, src_att, src_token_type), trg_label