"""Forward-only RNG diagnostic for observed recovery mismatch; no optimizer or training run."""
import os,sys
sys.dont_write_bytecode=True
os.environ['OMP_NUM_THREADS']='3'
from step3_common import *
from scripts.python import panel_pathformer_main_v2 as formal

def state_hash(state):return hashlib.sha256(state.numpy().tobytes()).hexdigest()
def main():
 torch.set_num_interop_threads(1)
 xd=torch.linspace(-2,2,32*90*5).reshape(32,90,5)
 xw=torch.linspace(-1,1,32*26*5).reshape(32,26,5)
 outputs={};rows=[]
 for n in [3,12]:
  torch.set_num_threads(n);formal.set_seed(42)
  model=formal._make_model('daily_weekly','adaptive')
  init=formal._state_dict_sha256(model);rng_init=state_hash(torch.get_rng_state())
  model.train()
  with torch.no_grad():
   trainout=model(xd,xw).numpy()
  rng_after=state_hash(torch.get_rng_state())
  model.eval()
  with torch.no_grad():evalout=model(xd,xw).numpy()
  outputs[n]=dict(train=trainout,eval=evalout)
  rows.append(dict(threads=n,seed=42,initial_state_sha256=init,rng_before=rng_init,rng_after_training_mode_forward=rng_after))
 result=dict(kind='forward-only synthetic-input diagnostic; no backward, optimizer or training epoch',results=rows,max_abs_train_mode_forward_delta=float(np.max(np.abs(outputs[3]['train']-outputs[12]['train']))),max_abs_eval_mode_forward_delta=float(np.max(np.abs(outputs[3]['eval']-outputs[12]['eval']))),historical_training_thread_count='NOT FOUND; 12 is a current-host comparison, not verified historical metadata')
 write_json(OUT/'thread_rng_diagnostic.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
