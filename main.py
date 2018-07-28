from experiments import train_test

if __name__=='__main__':
    dataset_names = ['mutag', 'DD', 'enzymes', 'NCI1', 'collab', 'imdb_action_romance',
                     "reddit_iama_askreddit_atheism_trollx", "reddit_multi_5K", "imdb_comedy_romance_scifi",'ptc']
    modes = ['vertex', 'edge', 'comb', 'vertex_channels']


    dataset = 'mutag'   # choose dataset frome dataset-list
    mode = modes[0]     # choose mode frome mode-list
    width = None        # None for default recommended values,
                        # for costume values: if 'comb' mode use tuple (vertex_width,edge_width)
                        #                     otherwise use integer
    k = 10              # common values: 5,10
    train_test(ds_name=dataset, K=k, mode=mode, W=width, max_epochs=50, test_percent=0.2, batch_size=20, savefig=False, showfig=True)