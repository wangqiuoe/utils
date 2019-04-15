#encoding:utf-8
import datetime
import json

def calculator(Q, period, rate, join_date, quit_date):
    quit_dt = datetime.datetime.strptime(quit_date, '%Y%m%d')
    join_dt = datetime.datetime.strptime(join_date, '%Y%m%d')
    time_interval_months = float((quit_dt - join_dt).days)/30
    run_month = [(join_dt + datetime.timedelta(days=30*i) ).strftime('%Y%m%d') for i in range(int(time_interval_months)+1)]
    money_plan = {
        join_date : Q
    }
    for i in run_month:
        this_period_money = money_plan[i]
        compound_interest = this_period_money*(rate*1.0/period)*(1.0/12)*sum(range(1, period+1))
        next_per_money = (this_period_money + compound_interest)/period
        for p in range(1, period+1):
            cur_month = (datetime.datetime.strptime(i, '%Y%m%d') + datetime.timedelta(days=30*p)).strftime('%Y%m%d')
            if cur_month not in money_plan:
                money_plan[cur_month] = 0
            money_plan[cur_month] += next_per_money
    quit_plan = []
    total_money = 0
    for k,v in sorted(money_plan.iteritems(), key = lambda x:x[0]):
        if k > quit_date:
            quit_plan.append([k,v])
            print '%s: %.2f' %(k, v)
            total_money += v
    print 'total_money: %.2f'  %(total_money)
    return quit_plan
            
if __name__ == "__main__" :
    Q = 16000
    period = 3
    rate = 0.13
    join_date = '20190105'
    quit_date = '20190316'
    quit_plan = calculator(Q, period, rate, join_date, quit_date)

