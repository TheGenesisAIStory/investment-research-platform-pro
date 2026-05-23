# quantdinger_mobile_summary_it

Generated: 2026-05-23T18:20:59.280350+00:00

Prepara una sintesi mobile breve e chiara.

Segnali:
date,symbol,name,sector,strategy_name,model_name,signal,target_weight,score,confidence,horizon,reasoning,status
2026-05-22,MSFT,Microsoft Corp.,Information Technology,risk_balanced_blend,synthetic_factor_baseline,1.0,0.2712292335548805,1.9929978018835703,0.7971991207534281,21D,"Blend di z-score prezzo, momentum a 63 sedute e penalita' per volatilita' recente.",paper
2026-05-22,SPY,SPDR S&P 500 ETF Trust,ETF,risk_balanced_blend,synthetic_factor_baseline,1.0,0.2561917040279175,1.4336515430705805,0.5734606172282322,21D,"Blend di z-score prezzo, momentum a 63 sedute e penalita' per volatilita' recente.",paper
2026-05-22,ISP.MI,Intesa Sanpaolo SpA,Financials,risk_balanced_blend,synthetic_factor_baseline,1.0,0.2543987628398716,1.4236182248152918,0.5694472899261167,21D,"Blend di z-score prezzo, momentum a 63 sedute e penalita' per volatilita' recente.",paper
2026-05-22,ASML.AS,ASML Holding NV,Information Technology,risk_balanced_blend,synthetic_factor_baseline,1.0,0.12369675126556576,0.692208356229097,0.2768833424916388,21D,"Blend di z-score prezzo, momentum a 63 sedute e penalita' per volatilita' recente.",paper
2026-05-22,AAPL,Apple Inc.,Information Technology,risk_balanced_blend,synthetic_factor_baseline,1.0,0.09448354831176464,0.5287309569445865,0.2114923827778346,21D,"Blend di z-score prezzo, momentum a 63 sedute e penalita' per volatilita' recente.",paper
2026-05-22,ENEL.MI,Enel SpA,Utilities,risk_balanced_blend,synthetic_factor_baseline,-1.0,0.0,-1.0654156739408065,0.4261662695763226,21D,"Blend di z-score prezzo, momentum a 63 sedute e penalita' per volatilita' recente.",paper
2026-05-22,NVDA,NVIDIA Corp.,Information Technology,risk_balanced_blend,synthetic_factor_baseline,-1.0,0.0,-0.626428004135635,0.250571201654254,21D,"Blend di z-score prezzo, momentum a 63 sedute e penalita' per volatilita' recente.",paper
2026-05-22,SAP.DE,SAP SE,Information Technology,risk_balanced_blend,synthetic_factor_baseline,-1.0,0.0,-0.24084881118435608,0.09633952447374243,21D,"Blend di z-score prezzo, momentum a 63 sedute e penalita' per volatilita' recente.",paper


Backtest:
backtest_id,strategy_name,model_name,universe,start_date,end_date,initial_capital,final_equity,total_return,annualized_return,annualized_volatility,sharpe,max_drawdown,avg_turnover,win_rate,metrics_json,created_at
1,risk_balanced_blend,synthetic_factor_baseline,"AAPL,MSFT,NVDA,SPY,ENEL.MI,ISP.MI,ASML.AS,SAP.DE",2021-01-04,2026-05-22,100000.0,209158.42460589568,1.0915842460589569,0.1415114668025219,0.18997186810043903,0.7916997826196022,-0.2231482448868355,0.20933782087328534,0.48612099644128115,"{""annualized_return"": 0.1415114668025219, ""annualized_volatility"": 0.18997186810043903, ""avg_turnover"": 0.20933782087328534, ""max_drawdown"": -0.2231482448868355, ""observations"": 1405.0, ""sharpe"": 0.7916997826196022, ""total_return"": 1.0915842460589569, ""win_rate"": 0.48612099644128115}",2026-05-23T18:18:53.938404+00:00


Output richiesto:
- 3 bullet massimo;
- una frase di rischio;
- una CTA prudente, ad esempio 'apri il laboratorio' o 'rivedi il backtest';
- niente promessa di rendimento.
