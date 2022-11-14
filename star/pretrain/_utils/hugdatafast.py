from functools import partial
from pathlib import Path
import json
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import datasets
from fastai.text.all import *


@delegates()
class MySortedDL(TfmdDL):
    "A :class:`DataLoader` that do smart batching and dynamic padding. Different from :class:`SortedDL`, it automatically pad every attribute of samples, is able to filter samples, and can be cached to sort/filter only at first time."

    def __init__(self, dataset, srtkey_fc=None, filter_fc=False, pad_idx=None, cache_file=None, **kwargs):
        """
        Args:
            dataset (HF_Dataset): Actually any object implements ``__len__`` and ``__getitem__`` that return a tuple as a sample.
            srtkey_fc (``*args->int``, optional): Get key for decending sorting from a sample .\n
              - If ``None``, sort by length of first element of a sample.
              - If ``False``, not sort. 
            filter_fc (``*args->bool``, optional): Return ``True`` to keep the sample.
            pad_idx (``int``, optional): pad each attribute of samples to the max length of its max length within the batch.\n 
              - If ``List[int]``, specify pad_idx for each attribute of a sample. e.g. a sample is a tuple (masked_inputs, labels), `pad_idx=[0 ,-100]` pad masked_inputs with 0, labels with -100.
              - If ``False``, do no padding. 
              - If ``None``, try ``dataset.pad_idx``, do no padding if no such attribute.
            cache_file (``str``, optional): Path of a json file to cache info for sorting and filtering.
            kwargs: key arguments for `TfmDl` or `DataLoader`
        Example:
            >>> samples = [ (torch.tensor([1]), torch.tensor([7,8]), torch.tensor(1)),,
            ...             (torch.tensor([2,3]), torch.tensor([9,10,11]), torch.tensor(2)),
            ...             (torch.tensor([4,5,6]), torch.tensor([11,12,13,14]), torch.tensor(3)), ]
            ... dl = MySortedDL(samples,
            ...                 srtkey_fc=lambda *args: len(args[0]),
            ...                 filter_fc=lambda x1,y1: y1<3,
            ...                 pad_idx=-1,
            ...                 cache_file='/tmp/cache.json', # calls after this will load cache
            ...                 bs=999, # other parameters go to `TfmDL` and `DataLoader`
            ...                 )
            ... dl.one_batch()
            (tensor([[ 2,  3],
                     [ 1, -1]]),
             tensor([[ 9, 10, 11],
                    [ 7,  8, -1]]),
             tensor([2, 1]))
        """
        # Defaults
        if srtkey_fc is not False: srtkey_fc = lambda *x: len(x[0])
        if pad_idx is None: pad_idx = getattr(dataset, 'pad_idx', False)
        if isinstance(pad_idx, int): pad_idxs = [pad_idx] * len(dataset[0])
        elif isinstance(pad_idx, (list, tuple)): pad_idxs = pad_idx
        cache_file = Path(cache_file) if cache_file else None
        idmap = list(range(len(dataset)))

        # Save attributes
        super().__init__(dataset, **kwargs)
        store_attr('pad_idxs,srtkey_fc,filter_fc,cache_file,idmap', self)

        # Prepare records for sorting / filtered samples
        if srtkey_fc or filter_fc:
          if cache_file and cache_file.exists():
            # load cache and check
            with cache_file.open(mode='r') as f: cache = json.load(f)
            idmap, srtkeys = cache['idmap'], cache['srtkeys']
            if srtkey_fc: 
              assert srtkeys, "srtkey_fc is passed, but it seems you didn't sort samples when creating cache."
              self.srtkeys = srtkeys
            if filter_fc:
              assert idmap, "filter_fc is passed, but it seems you didn't filter samples when creating cache."
              self.idmap = idmap
          else:
            # overwrite idmap if filter, get sorting keys if sort
            idmap = []; srtkeys = []
            for i in tqdm(range_of(dataset), leave=False):
                sample = self.do_item(i)
                if filter_fc and not filter_fc(*sample): continue
                if filter_fc: idmap.append(i)
                if srtkey_fc: srtkeys.append(srtkey_fc(*sample))
            if filter_fc: self.idmap = idmap
            if srtkey_fc: self.srtkeys = srtkeys
            # save to cache
            if cache_file:
              try: 
                with cache_file.open(mode='w+') as f: json.dump({'idmap':idmap,'srtkeys':srtkeys}, f)
              except: os.remove(str(cache_file))
          # an info for sorting
          if srtkey_fc: self.idx_max = np.argmax(self.srtkeys)
          # update number of samples
          if filter_fc: self.n = self.n = len(self.idmap)

    def create_item(self, i): return self.dataset[self.idmap[i]]

    def create_batch(self, samples):
        
        input_ids = [item[0] for item in samples]
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        sim_label = [item[1] for item in samples]
        sim_label = torch.tensor(sim_label, dtype=torch.float)
        sim_label.requires_grad = False
        sim_label_all = torch.zeros(sim_label.size(0), sim_label.size(0)).float()
        sim_label_all.requires_grad = False
        row,col,count = 0,0,0
        for idx,item in enumerate(sim_label):
            if idx != 0 and idx % 5 == 0:
                col += 5
            sim_label_all[row,col:col+5] = item
            row += 1
        # sim_label_all[:,-1] = 1.0
            
            
        
        sim_mask = [item[7] for item in samples]
        sim_mask = torch.tensor(sim_label, dtype=torch.float)
        sim_mask.requires_grad = False
        sim_mask_p = torch.zeros(sim_mask.size(0), sim_mask.size(0)).float()
        sim_mask_p.requires_grad = False
        sim_mask_n = torch.ones(sim_mask.size(0), sim_mask.size(0)).float()
        sim_mask_n.requires_grad = False
        row,col,count = 0,0,0
        for idx,item in enumerate(sim_mask):
            if idx != 0 and idx % 5 == 0:
                col += 5
            sim_mask_p[row,col:col+5] = (item > 0.0).float()
            sim_mask_n[row,col:col+5] = (item > 0.0).float()
            row += 1
        # sim_mask_p[:,-1] = True
        
        
        ssf_label = [col for item in samples for col in item[2]]
        ssf_label = torch.tensor(ssf_label, dtype=torch.long)


        question_mask_plm = [item[3] for item in samples]
        question_mask_plm = ~torch.tensor(question_mask_plm, dtype=torch.bool)
        question_mask_plm.requires_grad = False


        rtd_label = [item[8] for item in samples]
        rtd_label = torch.tensor(rtd_label, dtype=torch.bool)

        column_mask_plm = [item[4] for item in samples]
        column_mask_plm = torch.tensor(column_mask_plm, dtype=torch.bool)
        column_mask_plm.requires_grad = False

        column_word_num = [len(item[5]) for item in samples]
        themax = max(column_word_num)
        column_word_num = torch.tensor(column_word_num, dtype=torch.long)
        column_word_len = [item[5] + [0] * (themax - len(item[5])) for item in samples]
        column_word_len = torch.tensor(column_word_len, dtype=torch.long)

        position_ids = [item[6] for item in samples]
        position_ids = torch.tensor(position_ids, dtype=torch.long)


        return (input_ids, sim_label_all, ssf_label, question_mask_plm,  column_mask_plm, column_word_len, position_ids, column_word_num, sim_mask_p, sim_mask_n, rtd_label)



    def get_idxs(self):
        idxs = super().get_idxs()
        if self.shuffle: return idxs
        if self.srtkey_fc: return sorted(idxs, key=lambda i: self.srtkeys[i], reverse=True)
        return idxs

    def shuffle_fn(self,idxs):
        if not self.srtkey_fc: return super().shuffle_fn(idxs)
        idxs = np.random.permutation(self.n)
        idx_max = np.where(idxs==self.idx_max)[0][0]
        idxs[0],idxs[idx_max] = idxs[idx_max],idxs[0]
        sz = self.bs*50
        chunks = [idxs[i:i+sz] for i in range(0, len(idxs), sz)]
        chunks = [sorted(s, key=lambda i: self.srtkeys[i], reverse=True) for s in chunks]
        sort_idx = np.concatenate(chunks)

        sz = self.bs
        batches = [sort_idx[i:i+sz] for i in range(0, len(sort_idx), sz)]
        sort_idx = np.concatenate(np.random.permutation(batches[1:-1])) if len(batches) > 2 else np.array([],dtype=np.int)
        sort_idx = np.concatenate((batches[0], sort_idx) if len(batches)==1 else (batches[0], sort_idx, batches[-1]))
        return iter(sort_idx)

    @delegates(TfmdDL.new)
    def new(self, dataset=None, **kwargs):
        if 'get_idxs' in kwargs: # when Learner.get_preds, dataload has `get_idxs` will be cloned. So we need to prevent sorting again
          kwargs['cache_file'] = self.cache_file
        # We don't use filter_fc here cuz we can't don't validate certaion samples in dev/test set. 
        return super().new(dataset=dataset, pad_idx=self.pad_idx, srtkey_fc=self.srtkey_fc, filter_fc=False, **kwargs)

# =========================
#  Titled primitives
# =========================

class _Int(int, ShowPrint):
    def __new__(cls, *args, **kwargs):
        item = super().__new__(cls, *args)
        for n,v in kwargs.items(): setattr(item, n, v)
        return item

class _Float(float, ShowPrint):
    def __new__(cls, *args, **kwargs):
        item = super().__new__(cls, *args)
        for n,v in kwargs.items(): setattr(item, n, v)
        return item

class _Str(str, ShowPrint):
    def __new__(cls, *args, **kwargs):
        item = super().__new__(cls, *args)
        for n,v in kwargs.items(): setattr(item, n, v)
        return item

class _Tuple(fastuple, ShowPrint):
    def __new__(cls, *args, **kwargs):
        item = super().__new__(cls, *args)
        for n,v in kwargs.items(): setattr(item, n, v)
        return item 

class _L(L, ShowPrint):
    def __new__(cls, *args, **kwargs):
        item = super().__new__(cls, *args)
        for n,v in kwargs.items(): setattr(item, n, v)
        return item  

# only change "label" to "title"
def _show_title(o, ax=None, ctx=None, title=None, color='black', **kwargs):
    "Set title of `ax` to `o`, or print `o` if `ax` is `None`"
    ax = ifnone(ax,ctx)
    if ax is None: print(o)
    elif hasattr(ax, 'set_title'):
        t = ax.title.get_text()
        if len(t) > 0: o = t+'\n'+str(o)
        ax.set_title(o, color=color)
    elif isinstance(ax, pd.Series):
        while title in ax: title += '_'
        ax = ax.append(pd.Series({title: o}))
    return ax

class _ShowTitle:
    def show(self, ctx=None, **kwargs):
      kwargs['title'] = kwargs.pop('title', getattr(self, 'title', self.default_title))
      return _show_title(str(self), ctx=ctx, **kwargs)

# it seems that python prioritising prior inherited class when finding methods   

class _TitledInt(_ShowTitle, _Int): default_title = 'int'

class _TitledFloat(_ShowTitle, _Float): default_title = 'float'

# I created it, but it just print book likt int, haven't find a way to solve it
class _TitledBool(_ShowTitle, _Int): # python says bool can't be base class
    default_title = 'bool'

class _TitledStr(_ShowTitle, _Str):
    default_title = 'text'
    def truncate(self, n):
        "Truncate self to `n`"
        words = self.split(' ')[:n]
        return _TitledStr(' '.join(words), title=getattr(self, 'title', 'text'))

class _TitledTuple(_ShowTitle, _Tuple): default_title = 'list'

class _Category(_ShowTitle, _Str): default_title = 'label'

class _MultiCategory(_ShowTitle, _L):
    default_title = 'labels'
    def show(self, ctx=None, sep=';', color='black', **kwargs):
        kwargs['title'] = kwargs.pop('title', getattr(self, 'title', self.default_title))
        return _show_title(sep.join(self.map(str)), ctx=ctx, color=color, **kwargs)

""" Caution !!
These two function is inperfect.
But they cope with mutiple input columns problem (n_inp >1), which cause no df printing but just sequentail print
These will be a problem when you are doing non-text problem with n_inp > 1 (multiple input column),
which shouldn't be the case of huggingface/datasets user.
And I hope fastai come up with a good solution to show_batch multiple inputs problems for text/non-text.
"""
@typedispatch
def show_batch(x:tuple, y, samples, ctxs=None, max_n=9, **kwargs):
  if ctxs is None: ctxs = get_empty_df(min(len(samples), max_n))
  ctxs = show_batch[object](x, y, samples, max_n=max_n, ctxs=ctxs, **kwargs)
  display_df(pd.DataFrame(ctxs))
  return ctxs

@typedispatch
def show_results(x: tuple, y, samples, outs, ctxs=None, max_n=10, trunc_at=150, **kwargs):
  if ctxs is None: ctxs = get_empty_df(min(len(samples), max_n))
  ctxs = show_results[object](x, y, samples, outs, ctxs=ctxs, max_n=max_n, **kwargs)
  display_df(pd.DataFrame(ctxs))
  return ctxs

class HF_Dataset():
  """A wrapper for :class:`datasets.Dataset`.  It will behavior like original :class:`datasets.Dataset`, 
  but also function as a :class:`fastai.data.core.datasets` that provides samples and decodes."""
  
  def __init__(self, hf_dset, cols=None, hf_toker=None, neat_show=False, n_inp=1):
    """
    Args:
      hf_dset (:class:`datasets.Dataset`): Prerocessed Hugging Face dataset to be wrapped.
      cols (dict, optional): columns of :class:`datasets.Dataset` to be used to construct samples, and (optionally) semantic tensor type for each of those columns to decode.\n
        - cols(``Dict[Fastai Semantic Tensor]``): encode/decode column(key) with semantic tensor type(value). If {value} is ``noop``, semantic tensor of the column is by default `TensorTuple`.
        - cols(``list[str]``): specify only columns and take default setting for semantic tensor type of them.\n
          - if length is 1, regard the 1st element as `TensorText`
          - if length is 2, regard the 1st element as `TensorText`, 2nd element as `TensorCategory`
          - Otherwise, regard all elements as `TensorTuple`
        - cols(None): pass :data:`hf_dset.column_names` (list[str]) as cols.
      hf_toker (:class:`transformers.PreTrainedTokenizer`, optional): Hugging Face tokenizer, used in decode and provide ``pad_idx`` for dynamic padding
      neat_show (bool, optional): Show the original sentence instead of tokens joined by space.
      n_inp (int, optional): take the first ``n_inp`` columns of ``cols`` as x, and the rest as y .
    Example:
      >>> tokenized_cola_train_set[0]
      {'sentence': "Our friends won't buy this analysis, let alone the next one we propose.",
       'label': 1,
       'idx': 0,
       'text_idxs': [ 2256,  2814,  2180,  1005,  1056,  4965,  2023,  4106,  1010,  2292, 2894,  1996,  2279,  2028,  2057, 16599,  1012]}
      >>> hf_dset = HF_Datset(tokenized_cola_train_set, cols=['text_idxs', 'label'], hf_toker=tokenizer_electra_small_fast)
      >>> len(hf_dset), hf_dset[0]
      8551, (TensorText([ 2256,  2814,  2180,  1005,  1056,  4965,  2023,  4106,  1010,  2292, 2894,  1996,  2279,  2028,  2057, 16599,  1012]), TensorCategory(1))
      >>> hf_dset.decode(hf_dset[0])
      ("our friends won ' t buy this analysis , let alone the next one we propose .", '1')
      # The wrapped dataset "is" also the original huggingface dataset
      >>> hf_dset.column_names == tokenized_cola_train_set.column_names
      True
      # Manually specify `cols` with dict, here it is equivalent to the above. And addtionally, neatly decode samples.
      >>> neat_hf_dset = HF_Datset(tokenized_cola_train_set, {'text_idxs':TensorText, 'label':TensorCategory}, hf_toker=tokenizer_electra_small_fast, neat_show=True)
      >>> neat_hf_dset.decode(neat_hf_dset[0])
      ("our friends won't buy this analysis, let alone the next one we propose.", '1')
      # Note: Original set will be set to Pytorch format with columns specified in `cols`
      >>> tokenized_cola_train_set[0]
      {'label': tensor(1),
       'text_idxs': tensor([ 2256,  2814,  2180,  1005,  1056,  4965,  2023,  4106,  1010,  2292, 2894,  1996,  2279,  2028,  2057, 16599,  1012])}
    """
    
    # some default setting for tensor type used in decoding
    if cols is None: cols = hf_dset.column_names
    if isinstance(cols, list): 
      if n_inp==1: 
        if len(cols)==1: cols = {cols[0]: TensorText}
        elif len(cols)==2: cols = {cols[0]: TensorText, cols[1]: TensorCategory}
      else: cols = { c: noop for c in cols }
    assert isinstance(cols, dict)
    
    # make dataset output pytorch tensor
    # hf_dset.set_format( type='torch', columns=list(cols.keys()) )

    # store attributes
    self.pad_idx = hf_toker.pad_token_id
    self.hf_dset = hf_dset
    store_attr("cols,n_inp,hf_toker,neat_show", self)

  def __getitem__(self, idx):
    sample = self.hf_dset[idx]
    return tuple( tensor_cls(sample[col]) for col, tensor_cls in self.cols.items() )

  def __len__(self): return len(self.hf_dset)

  @property
  def col_names(self): return list(self.cols.keys())

  def decode(self, o, full=True): # `full` is for micmic `Dataset.decode` 
    if len(self.col_names) != len(o): return tuple( self._decode(o_) for o_ in o )
    return tuple( self._decode(o_, self.col_names[i]) for i, o_ in enumerate(o) )

  def _decode_title(self, d, title_cls, title): 
    if title: return title_cls(d, title=title)
    else: return title_cls(d)

  @typedispatch
  def _decode(self, t:torch.Tensor, title):
    if t.shape: title_cls = _TitledTuple
    elif isinstance(t.item(),bool): title_cls = _TitledBool # bool is also int, so check whether is bool first
    elif isinstance(t.item(),float): title_cls = _TitledFloat
    elif isinstance(t.item(),int): title_cls = _TitledInt
    return self._decode_title(t.tolist(), title_cls , title)

  @typedispatch
  def _decode(self, t:TensorText, title): 
    assert self.hf_toker, "You should give a huggingface tokenizer if you want to show batch."
    if self.neat_show: text = self.hf_toker.decode([idx for idx in t if idx != self.hf_toker.pad_token_id])
    else: text = ' '.join(self.hf_toker.convert_ids_to_tokens(t))
    return self._decode_title(text, _TitledStr, title)

  @typedispatch
  def _decode(self, t:LMTensorText, title): return self._decode[TensorText](self, t, title)

  @typedispatch
  def _decode(self, t:TensorCategory, title): return self._decode_title(t.item(), _Category, title)

  @typedispatch
  def _decode(self, t:TensorMultiCategory, title): return self._decode_title(t.tolist(), _MultiCategory, title)

  def __getattr__(self, name):
    "If not defined, let the datasets.Dataset in it act for us."
    if name in HF_Dataset.__dict__: return HF_Dataset.__dict__[name]
    elif name in self.__dict__: return self.__dict__[name]
    elif hasattr(self.hf_dset, name): return getattr(self.hf_dset, name)
    raise AttributeError(f"Both 'HF_Dataset' object and 'datasets.Dataset' object have no '{name}' attribute ")
  
class HF_Datasets(FilteredBase):
  """Function as :class:`fastai.data.core.Datasets` to create :class:`fastai.data.core.Dataloaders` from a group of :class:`datasets.Dataset`s"""

  _dl_type,_dbunch_type = MySortedDL,DataLoaders
  
  @delegates(HF_Dataset.__init__)
  def __init__(self, hf_dsets: dict, test_with_y=False, **kwargs):
    """
    Args:
      hf_dsets (`Dict[datasets.Dataset]`): Prerocessed Hugging Face Datasets, {key} is split name, {value} is :class:`datasets.Dataset`, order will become the order in :class:`fastai.data.core.Dataloaders`.
      test_with_y (bool, optional): Whether the test set come with y (answers) but not with fake y (e.g. all -1 label). 
        If ``False``, tell only test set to construct samples from first ``n_inp`` columns (do not output fake y). 
        And all datasets passed in ``hf_dsets`` with its name starts with "test" will be regarded as test set. 
      kwargs: Passed to :class:`HF_Dataset`. Be sure to pass arguments that :class:`HF_Dataset` needs !!
    """
    cols, n_inp = kwargs.pop('cols', None), kwargs.get('n_inp', 1)
    self.hf_dsets = {};
    for split, dset in hf_dsets.items():
      if cols is None: cols = dset.column_names
      if split.startswith('test') and not test_with_y: 
        if isinstance(cols, list): _cols = cols[:n_inp]
        else: _cols = { k:v for _, (k,v) in zip(range(n_inp),cols.items()) }
      else: _cols = cols
      self.hf_dsets[split] = HF_Dataset(dset, cols=_cols, **kwargs)

  def subset(self, i): return list(self.hf_dsets.values())[i]
  def __getitem__(self, split): return self.hf_dsets[split]
  @property
  def n_subsets(self): return len(self.hf_dsets)
  @property
  def cache_dir(self): return Path(next(iter(self.hf_dsets.values())).cache_files[0]['filename']).parent
  
  @delegates(FilteredBase.dataloaders)
  def dataloaders(self, device='cpu', cache_dir=None, cache_name=None, dl_kwargs=None, **kwargs):
    """
    Args:
      device (str): device where outputed batch will be on. Because a batch will be loaded to test when creating :class: `fastai.data.core.Dataloaders`, to prevent always leaving a batch of tensor in cuda:0, using default value cpu and then ``dls.to(other device)`` at the time you want is suggested.
      cache_dir (str, optional): directory to store caches of :class:`MySortedDL`. if ``None``, use cache directory of the first :class:`datasets.Dataset` in ``hf_dsets`` that passed to :method:`HF_Datasets.__init__`.
      cache_name (str, optional): format string that includes one param "{split}", which will be replaced with name of split as cache file name under `cache_dir` for each split. If ``None``, tell :class:MySortedDL don't do caching.
      dl_kwargs (list[dict], optional): ith item is addtional kwargs to be passed to initialization of ith dataloader for ith split
      kwargs: Passed to :func:`fastai.data.core.FilteredBase.dataloaders`
    
    Example:
      >>> tokenized_cola
      {'train': datasets.Dataset, 'validation': datasets.Dataset, 'test': datasets.Dataset}
      >>> tokenized_cola['test'][0]
      {'sentence': 'Bill whistled past the house.',
       'label': -1, # Fake label. True labels are not open to the public.
       'idx': 0,
       'text_idxs': [3021, 26265, 2627, 1996, 2160, 1012]}
      >>> dls = HF_Datasets(tokenized_cola,
      ...                   cols=['text_idxs', 'label'], hf_toker=hf_tokenizer,  # args for HF_Dataset
      ...                   ).dataloaders(bs=32 , cache_name="dl_cached_for_{split}") # args for MySortedDL
      >>> dls.show_batch(max_n=2)
                                                                                                                         text_idxs           label
      ---------------------------------------------------------------------------------------------------------------------------------------------
      0  everybody who has ever, worked in any office which contained any typewriter which had ever been used to type any letters which had    1
         to be signed by any administrator who ever worked in any department like mine will know what i mean.
      ---------------------------------------------------------------------------------------------------------------------------------------------
      1  playing with matches is ; lots of fun, but doing, so and emptying gasoline from one can to another at the same time is a sport best   1
         reserved for arsons.
      # test set won't produce label becuase of `test_with_y=False`   
      >>> dls[-1].show_batch(max_n=2) 
                                                                                     text_idxs
      ------------------------------------------------------------------------------------------
      0  cultural commissioner megan smith said that the five ` ` soundscape'' pieces would ` `
         give a festive air to park square, they're fun and interesting''.
      ------------------------------------------------------------------------------------------
      1  wendy is eager to sail around the world and bruce is eager to climb kilimanjaro, but 
         neither of them can because money is too tight.
    """
    if dl_kwargs is None: dl_kwargs = [{} for _ in range(len(self.hf_dsets))]
    elif isinstance(dl_kwargs, dict):
      dl_kwargs = [ dl_kwargs[split] if split in dl_kwargs else {} for split in self.hf_dsets]
    # infer cache file names for each dataloader if needed
    dl_type = kwargs.pop('dl_type', self._dl_type)
    if dl_type==MySortedDL and cache_name:
      assert "{split}" in cache_name, "`cache_name` should be a string with '{split}' in it to be formatted."
      cache_dir = Path(cache_dir) if cache_dir else self.cache_dir
      cache_dir.mkdir(exist_ok=True)
      if not cache_name.endswith('.json'): cache_name += '.json'
      for i, split in enumerate(self.hf_dsets):
        filled_cache_name = dl_kwargs[i].pop('cache_name', cache_name.format(split=split))
        if 'cache_file' not in dl_kwargs[i]:
          dl_kwargs[i]['cache_file'] = cache_dir/filled_cache_name
    # change default to not drop last
    kwargs['drop_last'] = kwargs.pop('drop_last', False)
    # when corpus like glue/ax has only testset, set it to non-train setting
    if list(self.hf_dsets.keys())[0].startswith('test'):
      kwargs['shuffle_train'] = False
      kwargs['drop_last'] = False
    return super().dataloaders(dl_kwargs=dl_kwargs, device=device, **kwargs)