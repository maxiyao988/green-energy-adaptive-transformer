"""Original deterministic four-alpha Ridge protocol, post-neural-training only."""
import sys,pickle
sys.dont_write_bytecode=True
import io_g as io
import worker_g as w
from scripts.python import panel_baseline_ridge as ridge
np,pd=w.old.np,w.old.pd

def main():
    io.barrier();ds=w.dataset(True);stats=w.old.common.fit_train_only_normalizer(ds);X=w.old.common.flatten_features('daily_weekly',w.old.common.apply_normalizer(ds,stats,'daily_weekly'));tr=ds.split=='train';val=ds.split=='val'
    for h in io.HS:
        parent=io.OUT/'ridge'/f'h{h}';done=list(parent.glob('attempt_*/COMPLETE.json'))
        if done:assert len(done)==1;io.verify_record(done[0]);continue
        folder=io.target(parent/('attempt_'+io.stamp()));folder.mkdir(parents=True);models=[];original=ridge.Ridge
        def capture(*a,**kw):
            m=original(*a,**kw);models.append(m);return m
        ridge.Ridge=capture
        try:summary,formal_test=ridge.run_one(ds,stats,'daily_weekly',h)
        finally:ridge.Ridge=original
        assert [m.alpha for m in models]==[.1,1.,10.,100.]
        chosen=next(m for m in models if m.alpha==summary['best_alpha'])
        cp=io.target(folder/'best_model.pkl')
        with cp.open('xb') as f:pickle.dump(chosen,f,protocol=5)
        trials=[]
        for m in models:trials.append(dict(h=h,alpha=m.alpha,val_mse=float(np.mean((m.predict(X[val])-ds.y[h][val])**2)),selected=m.alpha==summary['best_alpha'],evidence='EXPLORATORY descriptive'))
        io.csv(folder/'validation_trials.csv',pd.DataFrame(trials))
        for sp in ['train','val']:
            mask=ds.split==sp;frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=pd.to_datetime(ds.anchor_date[mask]).strftime('%Y-%m-%d'),split=sp,y_true=ds.y[h][mask],y_pred=chosen.predict(X[mask])));io.csv(folder/(sp+'.csv'),frame)
        mask=ds.split=='test';frame=pd.DataFrame(dict(ticker=ds.ticker[mask],anchor_date=ds.anchor_date[mask],split='test',y_true=ds.y[h][mask],y_pred=chosen.predict(X[mask])))
        assert np.array_equal(frame.y_pred.to_numpy(),formal_test.y_pred.to_numpy())
        job=dict(setting='T3',model='ridge',variant='Ridge-CS',h=h,seed=-1,gate=False,window='W3',learning_rate=None,delta=io.read(io.OUT/'SIGMAS_CS.json')['values'][str(h)],target='cross_sectional',evidence='EXPLORATORY')
        w.saved_evaluation(job,folder,ds,frame,io.rel(cp));io.verify_record(folder/'COMPLETE.json');print('RIDGE-CS h',h,'COMPLETE',flush=True)
if __name__=='__main__':main()
