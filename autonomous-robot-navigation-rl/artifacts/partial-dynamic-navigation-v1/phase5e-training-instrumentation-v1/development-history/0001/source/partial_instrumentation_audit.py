"""Independent numerical checks of already-captured live tensors; no model calls."""
import numpy as np

from .partial_instrumentation_trace import array


def near(actual,expected,tolerance=2e-5):
    if not np.allclose(actual,expected,atol=tolerance,rtol=2e-6):
        raise ValueError('Actual captured numerical computation disagrees with independent formula')


def validate_dqn(row):
    r=array(row['reward']);done=array(row['consumed_dones']);discount=array(row['discount'])
    nxt=array(row['next_target_max_Q']);current=array(row['current_chosen_Q']);target=array(row['actual_target'])
    near(target,r+(1-done)*discount*nxt)
    error=target.astype(float)-current.astype(float);absolute=np.abs(error)
    loss=np.where(absolute<1,.5*error**2,absolute-.5).mean()
    near(array(row['loss']),loss)
    for s,d in zip(row['sample'],done.reshape(-1)):
        expected=float(s['terminated'])
        if d!=expected:raise ValueError('DQN timeout mask/transition identity mismatch')
    return {'rows':len(r),'loss_error':abs(float(array(row['loss']).item())-float(loss)),
        'target_max_error':float(np.max(np.abs(target-(r+(1-done)*discount*nxt))))}


def validate_gae(inputs,result):
    if inputs['rollout_id']!=result['rollout_id'] or not inputs.get('rows'):
        raise ValueError('Missing bootstrap or mismatched rollout identity')
    values=array(inputs['values']).astype(float);rewards=array(inputs['rewards']).astype(float)
    starts=array(inputs['episode_starts']);last=array(inputs['last_values']).reshape(-1)
    dones=array(inputs['dones']);gamma=inputs['gamma'];lam=inputs['gae_lambda']
    computed=np.zeros_like(values);carry=np.zeros(values.shape[1])
    for i in reversed(range(len(values))):
        mask=1-dones if i==len(values)-1 else 1-starts[i+1]
        next_v=last if i==len(values)-1 else values[i+1]
        delta=rewards[i]+gamma*mask*next_v-values[i]
        carry=delta+gamma*lam*mask*carry;computed[i]=carry
    actual=array(result['advantages']);near(actual,computed)
    near(array(result['returns']),actual+values)
    return {'rows':len(values),'advantage_max_error':float(np.max(np.abs(actual-computed)))}


def validate_ppo(row):
    if row.get('Q_values')!='not_applicable':raise ValueError('PPO cannot have Q-values')
    raw=array(row['raw_advantages']).astype(float);adv=array(row['normalized_advantages'])
    stats=row['normalization_statistics']
    if 'mean' in stats:
        near(array(stats['mean']),raw.mean());near(array(stats['std']),raw.std(ddof=1))
        near(adv,(raw-raw.mean())/(raw.std(ddof=1)+1e-8))
    else:near(adv,raw)
    old=array(row['old_log_probs']);new=array(row['new_log_probs']);ratio=array(row['ratio'])
    near(ratio,np.exp(new.astype(float)-old.astype(float)))
    c=row['clip_range'];first=adv*ratio;second=adv*np.clip(ratio,1-c,1+c)
    near(array(row['policy_loss_1']),first);near(array(row['policy_loss_2']),second)
    pg=-np.minimum(first,second).mean();vl=((array(row['returns'])-array(row['values_pred']))**2).mean()
    ent=-array(row['entropy']).mean() if row['entropy'] is not None else new.mean()
    near(array(row['policy_loss']),pg);near(array(row['value_loss']),vl);near(array(row['entropy_loss']),ent)
    total=pg+row['ent_coef']*ent+row['vf_coef']*vl
    near(array(row['total_loss']),total)
    logratio=new.astype(float)-old.astype(float);near(array(row['approx_kl']),(np.exp(logratio)-1-logratio).mean())
    near(row['clip_fraction'],(np.abs(ratio-1)>c).mean())
    return {'rows':len(raw),'total_loss_error':abs(float(array(row['total_loss']).item())-float(total)),
        'early_stop':row['early_stop']}


def validate_linear(row):
    before=array(row['weights_before']);after=array(row['weights_after']);phi=array(row['features'])
    action=row['action'];target=row['reward']
    if not row['terminated']:
        nxt=array(row['next_values'])
        target+=row['gamma']*(nxt[row['sampled_next_action']] if row['sampled_next_action'] is not None else nxt.max())
    near(row['target'],target,1e-10);error=target-before[action]@phi;near(row['td_error'],error,1e-10)
    expected=before.copy();expected[action]+=row['alpha']*error*phi/(phi@phi)
    near(after,expected,1e-10);near(row['update_norm'],np.linalg.norm(after-before),1e-10)
    return {'target_error':abs(target-row['target']),'weight_max_error':float(np.max(np.abs(after-expected)))}


def audit_records(rows):
    result={'DQN':[],'PPO':[],'GAE':[],'linear':[]};gae={};optimizers=[];permutations={};seen_losses=[]
    for row in rows:
        event=row['event']
        if event=='dqn_loss':result['DQN'].append(validate_dqn(row));seen_losses.append(row)
        elif event=='ppo_loss':result['PPO'].append(validate_ppo(row));seen_losses.append(row)
        elif event=='gae_input':gae[row['rollout_id']]=row
        elif event=='gae_result':
            if row['rollout_id'] not in gae:raise ValueError('GAE without actual pre-update bootstrap')
            result['GAE'].append(validate_gae(gae[row['rollout_id']],row))
        elif event=='linear_update':result['linear'].append(validate_linear(row))
        elif event=='optimizer':
            if row['update_id']!=len(optimizers)+1:raise ValueError('Non-monotonic optimizer ID')
            loss=next((r for r in reversed(seen_losses) if r.get('optimizer_step',r.get('update_id'))==row['update_id']),None)
            if loss is None or loss['network']!=row['network_before']:raise ValueError('Mismatched actual network version')
            grads=[array(g).astype(float) for g in row['gradient_tensors']]
            near(row['clipped_gradient_norm'],np.sqrt(sum(np.sum(g*g) for g in grads)),1e-10)
            optimizers.append(row)
        elif event=='ppo_permutation':
            indices=array(row['indices']).tolist()
            if sorted(indices)!=list(range(len(indices))):raise ValueError('Malformed PPO permutation')
            permutations[row['rollout_id'],row['epoch']]=indices
        elif event=='rollout_sample':
            key=row['rollout_id'],row['epoch']
            if key not in permutations or any(s['rollout_id']!=row['rollout_id'] for s in row['sample']):
                raise ValueError('Mismatched rollout sample identity')
        elif event=='ppo_timeout_bootstrap':
            added=next((r for r in rows if r['event']=='rollout_add' and r['transition_id']==row['transition_id']),None)
            if added is None:raise ValueError('Timeout augmentation missing consumed rollout row')
            near(array(added['adjusted_reward']),row['raw_reward']+row['gamma']*array(row['terminal_value']))
    return {'status':'passed','counts':{k:len(v) for k,v in result.items()},'details':result,'optimizer_steps':len(optimizers)}
