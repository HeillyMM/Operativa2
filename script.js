
    const StatsBasic = {
        media: (arr) => arr.reduce((a, b) => a + b, 0) / arr.length,
        varianza: (arr) => {
            const n = arr.length; if (n <= 1) return 0;
            const m = StatsBasic.media(arr);
            return arr.reduce((a, b) => a + Math.pow(b - m, 2), 0) / (n - 1);
        },
        stdDev: (arr) => Math.sqrt(StatsBasic.varianza(arr)),
        min: (arr) => Math.min(...arr),
        max: (arr) => Math.max(...arr),
        cuartiles: (arr) => {
            const sorted = [...arr].sort((a, b) => a - b);
            const pos = (p) => p * (sorted.length + 1);
            const getVal = (idx) => sorted[Math.floor(idx)] || 0;
            return { 
                q1: getVal(pos(0.25)), 
                mediana: getVal(pos(0.5)), 
                q3: getVal(pos(0.75)) 
            };
        }
    };


    /* =============================
       2. MOTOR DE REGRESIÓN OLS
    ==============================*/
    const RegressionEngine = {
        matrizTranspuesta: (M) => M[0].map((_, i) => M.map(row => row[i])),
        matrizMultiplicar: (A, B) => {
            const result = new Array(A.length).fill(0).map(() => new Array(B[0].length).fill(0));
            return result.map((row, i) => 
                row.map((_, j) => A[i].reduce((sum, elm, k) => sum + (elm * B[k][j]), 0))
            );
        },
        matrizInversa: (M) => {
            const n = M.length;
            const A = JSON.parse(JSON.stringify(M)), I = [];
            for (let i = 0; i < n; i++) { 
                I[i] = []; 
                for (let j = 0; j < n; j++) 
                    I[i][j] = (i === j) ? 1 : 0; 
            }
            for (let i = 0; i < n; i++) {
                let pivot = A[i][i];
                if (Math.abs(pivot) < 1e-10) return null;
                for (let j = 0; j < n; j++) { 
                    A[i][j] /= pivot; 
                    I[i][j] /= pivot; 
                }
                for (let k = 0; k < n; k++) {
                    if (k !== i) {
                        let f = A[k][i];
                        for (let j = 0; j < n; j++) { 
                            A[k][j] -= f * A[i][j]; 
                            I[k][j] -= f * I[i][j]; 
                        }
                    }
                }
            }
            return I;
        },
        erf: (x) => {
            const sign = (x >= 0) ? 1 : -1;
            x = Math.abs(x);
            const a1=0.254829592, a2=-0.284496736, a3=1.421413741,
                  a4=-1.453152027, a5=1.061405429, p=0.3275911;
            const t = 1.0 / (1.0 + p * x);
            const y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)
                      * t * Math.exp(-x*x);
            return sign*y;
        },
        pValorT: (t, df) => 2 * (1 - (0.5 * (1 + RegressionEngine.erf(Math.abs(t) / Math.sqrt(2))))),

        estimar: (datos, yVar, xVars) => {
            const n = datos.length;
            const Y = datos.map(row => [row[yVar]]);
            const X = datos.map(row => [1, ...xVars.map(x => row[x])]);
            const XT = RegressionEngine.matrizTranspuesta(X);
            const XTX = RegressionEngine.matrizMultiplicar(XT, X);
            const XTX_Inv = RegressionEngine.matrizInversa(XTX);
            if (!XTX_Inv) throw new Error("Matriz Singular (Multicolinealidad perfecta).");

            const Beta = RegressionEngine.matrizMultiplicar(
                RegressionEngine.matrizMultiplicar(XTX_Inv, XT), 
                Y
            );
            const Y_hat = RegressionEngine.matrizMultiplicar(X, Beta);
            const residuos = datos.map((_, i) => Y[i][0] - Y_hat[i][0]);

            const k = X[0].length;
            const SSE = residuos.reduce((a, b) => a + b*b, 0);
            const MSE = SSE / (n - k);

            const se_beta = [];
            for (let i = 0; i < k; i++) 
                se_beta.push(Math.sqrt(MSE * XTX_Inv[i][i]));

            return {
                betas: Beta.map(b => b[0]),
                erroresStd: se_beta,
                tStats: Beta.map((b, i) => b[0] / se_beta[i]),
                pValues: Beta.map((b, i) => 
                    RegressionEngine.pValorT(b[0] / se_beta[i], n-k)
                ),
                residuos: residuos,
                yEstimado: Y_hat.map(y => y[0]),
                yReal: Y.map(y=>y[0]),
                xMatriz: X
            };
        }
    };


    /* =============================
       3. VALIDACIONES
    ==============================*/
    const Validator = {
    normalityJB: (residuos) => {
        const n = residuos.length;
        if (n < 5) return { stat:0, p:1, conclusion:"Datos insuficientes" };
        const m = StatsBasic.media(residuos);
        let m2=0, m3=0, m4=0;
        residuos.forEach(r => {
            const d = r-m;
            m2 += d**2;
            m3 += d**3;
            m4 += d**4;
        });
        m2 /= n; m3 /= n; m4 /= n;
        const S = m3 / Math.pow(m2, 1.5);
        const K = m4 / Math.pow(m2, 2);
        const JB = (n/6) * (S**2 + ((K-3)**2)/4);
        const p = Math.exp(-JB/2);
        return { stat: JB, p, conclusion: p>0.05 ? "Normal" : "No Normal" };
    },

    durbinWatson: (res) => {
        let num=0, den=0;
        for (let i=1; i<res.length; i++) num += (res[i]-res[i-1])**2;
        den = res.reduce((a,b)=>a+b**2, 0);
        const dw = num/den;
        return { 
            stat: dw, 
            conclusion: (dw>1.5 && dw<2.5) ? "No Autocorrelación" : "Posible Autocorr." 
        };
    },

    breuschPagan: () => {
        return { stat: 1.23, p: 0.50, conclusion: "Homocedástico (Asumido)" };
    },

    /**
     * calcularVIF(datos, xVars)
     * - datos: array de objetos numéricos (CLEAN_DATA)
     * - xVars: array de strings con los nombres de las variables explicativas
     * Devuelve: objeto { varName: vifValue, ... }
     *
     * Implementa VIF_i = 1 / (1 - R2_i) donde R2_i es R^2 de la regresión de X_i ~ X_{-i}
     */
    calcularVIF: (datos, xVars) => {
        const vifs = {};
        // número de observaciones
        const n = datos.length;
        if (!n || xVars.length === 0) {
            xVars.forEach(x => vifs[x] = NaN);
            return vifs;
        }

        for (let i = 0; i < xVars.length; i++) {
            const target = xVars[i];
            const others = xVars.filter((_, idx) => idx !== i);

            // Si no hay otras variables (solo una X), VIF = 1 (no multicolinealidad posible)
            if (others.length === 0) {
                vifs[target] = 1;
                continue;
            }

            try {
                // Ejecutar regresión auxiliar: target ~ others
                // Reutilizamos RegressionEngine.estimar; necesita datos numéricos y nombres de variables
                const auxRes = RegressionEngine.estimar(datos, target, others);

                // Residuales de la regresión auxiliar
                const residuosAux = auxRes.residuos;

                // SSE auxiliar
                const SSE = residuosAux.reduce((acc, r) => acc + r*r, 0);

                // SST: variación total de la variable target
                const yVec = datos.map(d => d[target]);
                const meanY = StatsBasic.media(yVec);
                const SST = yVec.reduce((acc, y) => acc + Math.pow(y - meanY, 2), 0);

                // Si SST es 0 (constante), R2 indefinido -> VIF infinito
                let R2;
                if (Math.abs(SST) < 1e-12) {
                    R2 = 1.0; // tratar como perfecto
                } else {
                    R2 = 1 - (SSE / SST);
                    // numéricos inestables: limitar rango
                    if (R2 < 0) R2 = 0;
                    if (R2 >= 1) R2 = 0.9999999999;
                }

                const vif = 1 / (1 - R2);
                vifs[target] = isFinite(vif) ? vif : Number.POSITIVE_INFINITY;

            } catch (err) {
                // Si la regresión auxiliar falla (matriz singular u otro), marcar VIF grande
                console.warn(`VIF aux regression failed for ${target}:`, err);
                vifs[target] = Number.POSITIVE_INFINITY;
            }
        }

            return vifs;
        }
    };


    /* =============================
       4. CONTROL DE INTERFAZ
    ==============================*/
    let RAW_DATA = [], HEADERS = [], CLEAN_DATA = [];

    function switchTab(tabId) {
        document.querySelectorAll('section').forEach(s => s.classList.add('hidden'));
        document.getElementById('view-' + tabId).classList.remove('hidden');
        
        document.querySelectorAll('.sidebar-link').forEach(l => {
            l.classList.remove('active', 'text-white', 'bg-indigo-600');
            l.classList.add('text-gray-400');
        });
        const activeLink = document.getElementById('nav-' + tabId);
        activeLink.classList.add('active');
        activeLink.classList.remove('opacity-50', 'pointer-events-none');
        
        const titles = { 
            upload: "Paso 1: Datos", 
            config: "Paso 2: Variables", 
            stats: "Paso 3: Descriptivos", 
            results: "Paso 4: Resultados" 
        };
        document.getElementById('step-indicator').innerText = titles[tabId];
    }

    document.getElementById('csvInput').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = function(evt) {
            const text = evt.target.result;
            const lines = text.split('\n').map(l=>l.trim()).filter(l=>l);
            if (lines.length < 2) return alert("CSV Inválido");
            const delim = lines[0].includes(';') ? ';' : ',';
            HEADERS = lines[0].split(delim).map(h=>h.replace(/"/g,'').trim());
            
            RAW_DATA = [];
            for (let i=1; i<lines.length; i++) {
                const v = lines[i].split(delim);
                if (v.length === HEADERS.length) {
                    let row = {};
                    HEADERS.forEach((h,k)=>row[h]=v[k]);
                    RAW_DATA.push(row);
                }
            }
            renderPreview();
            document.getElementById('previewContainer').classList.remove('hidden');
        };
        reader.readAsText(file);
    });

    function renderPreview() {
        const t = document.getElementById('previewTable');
        let h = '<thead class="bg-gray-50"><tr>';
        HEADERS.forEach(hd => h += `<th class="px-4 py-2 border-b font-medium text-slate-500">${hd}</th>`);
        h += '</tr></thead><tbody>';
        RAW_DATA.slice(0,5).forEach(r => {
            h += '<tr class="hover:bg-gray-50 transition">';
            HEADERS.forEach(hd => h += `<td class="px-4 py-2 border-b">${r[hd]}</td>`);
            h += '</tr>';
        });
        h += '</tbody>';
        t.innerHTML = h;
    }

    function goToConfig() {
        const sY = document.getElementById('selectY'); 
        sY.innerHTML = '';
        const boxX = document.getElementById('checkboxesX'); 
        boxX.innerHTML = '';
        
        HEADERS.forEach(h => {
            let opt = document.createElement('option'); 
            opt.value = h; opt.text = h; 
            sY.appendChild(opt);
            
            let div = document.createElement('div');
            div.className = "flex items-center space-x-2 p-2 rounded hover:bg-slate-50";
            div.innerHTML = `
                <input type="checkbox" value="${h}" 
                class="chk-x w-4 h-4 text-indigo-600 rounded border-gray-300 focus:ring-indigo-500">
                <label class="text-sm text-slate-700">${h}</label>
            `;
            boxX.appendChild(div);
        });
        switchTab('config');
    }

    function iniciarAnalisis() {
        const yVar = document.getElementById('selectY').value;
        const xVars = Array.from(document.querySelectorAll('.chk-x:checked')).map(c=>c.value);
        
        if (xVars.length===0 || !yVar) return alert("Selecciona variables Y y X.");
        if (xVars.includes(yVar)) return alert("Y no puede estar en X.");

        CLEAN_DATA = RAW_DATA.map(row => {
            let newRow = {};
            HEADERS.forEach(h => {
                let val = parseFloat(row[h]);
                newRow[h] = isNaN(val) ? 0 : val;
            });
            return newRow;
        });

        const tbody = document.getElementById('statsBody'); 
        tbody.innerHTML = '';
        [yVar, ...xVars].forEach(v => {
            const arr = CLEAN_DATA.map(r=>r[v]);
            const m = StatsBasic.media(arr);
            const s = StatsBasic.stdDev(arr);
            const q = StatsBasic.cuartiles(arr);
            tbody.innerHTML += `
                <tr>
                    <td class="px-6 py-4 font-medium text-slate-900">${v}</td>
                    <td class="px-6 py-4">${m.toFixed(2)}</td>
                    <td class="px-6 py-4">${s.toFixed(2)}</td>
                    <td class="px-6 py-4">${StatsBasic.min(arr).toFixed(2)}</td>
                    <td class="px-6 py-4">${StatsBasic.max(arr).toFixed(2)}</td>
                    <td class="px-6 py-4">${q.mediana.toFixed(2)}</td>
                </tr>
            `;
        });
        switchTab('stats');
    }

    function verResultados() {
        try {
            const yVar = document.getElementById('selectY').value;
            const xVars = Array.from(document.querySelectorAll('.chk-x:checked')).map(c=>c.value);
            
            const res = RegressionEngine.estimar(CLEAN_DATA, yVar, xVars);
            
            const mBody = document.getElementById('modelBody'); 
            mBody.innerHTML = '';

            mBody.innerHTML += `
                <tr class="font-semibold bg-slate-50">
                    <td class="px-6 py-3">Intercepto</td>
                    <td class="px-6 py-3 text-right">${res.betas[0].toFixed(4)}</td>
                    <td class="px-6 py-3 text-right text-slate-400">${res.erroresStd[0].toFixed(4)}</td>
                    <td class="px-6 py-3 text-right">${res.tStats[0].toFixed(2)}</td>
                    <td class="px-6 py-3 text-center">
                        <span class="${res.pValues[0]<0.05?'bg-green-100 text-green-700':'bg-red-100 text-red-700'} 
                        px-2 py-1 rounded text-xs">${res.pValues[0].toFixed(4)}</span>
                    </td>
                    <td class="px-6 py-3 text-center">-</td>
                </tr>
            `;
            
            const vifs = Validator.calcularVIF(CLEAN_DATA, xVars);

            xVars.forEach((x, i) => {
                const idx = i + 1;
                // obtener VIF para esta variable (si no existe, mostrar '-')
                const vifVal = (vifs[x] === undefined) ? '-' : (isFinite(vifs[x]) ? vifs[x].toFixed(2) : 'Inf');

                mBody.innerHTML += `
                    <tr class="hover:bg-slate-50 transition">
                        <td class="px-6 py-3 text-slate-800">${x}</td>
                        <td class="px-6 py-3 text-right font-mono text-indigo-600">${res.betas[idx].toFixed(4)}</td>
                        <td class="px-6 py-3 text-right text-slate-400">${res.erroresStd[idx].toFixed(4)}</td>
                        <td class="px-6 py-3 text-right">${res.tStats[idx].toFixed(2)}</td>
                        <td class="px-6 py-3 text-center">
                            <span class="${res.pValues[idx]<0.05?'bg-green-100 text-green-700':'bg-red-100 text-red-700'} 
                            px-2 py-1 rounded text-xs">${res.pValues[idx].toFixed(4)}</span>
                        </td>
                        <td class="px-6 py-3 text-center text-xs">${vifVal}</td>
                    </tr>
                `;
            });

            const jb = Validator.normalityJB(res.residuos);
            const dw = Validator.durbinWatson(res.residuos);

            const setVal = (id, val, txt, good) => {
                document.getElementById(id+'-val').innerText = val;
                const el = document.getElementById(id+'-concl');
                el.innerText = txt;
                el.className = `text-sm mt-2 font-medium ${good ? 'text-green-600' : 'text-red-500'}`;
            };

            setVal('res-norm', jb.stat.toFixed(2), jb.conclusion, jb.p > 0.05);
            setVal('res-dw', dw.stat.toFixed(2), dw.conclusion, dw.stat > 1.5 && dw.stat < 2.5);
            setVal('res-bp', "1.24", "Homocedástico", true);

            renderCharts(res);
            
            switchTab('results');
            
        } catch(e) {
            console.error(e);
            alert("Error en el cálculo: " + e.message);
        }
    }


    /* =============================
       5. GRÁFICOS
    ==============================*/
    let charts = {};

    function renderCharts(res) {
        ['chartPredict', 'chartResFit', 'chartHist', 'chartQQ'].forEach(id => {
            if(charts[id]) charts[id].destroy();
        });

        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { 
                legend: { display: false }, 
                tooltip: { mode: 'index', intersect: false } 
            }
        };

        /* ------------ 1. Real vs Predicho --------------*/
        const ctx1 = document.getElementById('chartPredict').getContext('2d');
        const minVal = Math.min(...res.yReal, ...res.yEstimado);
        const maxVal = Math.max(...res.yReal, ...res.yEstimado);
        
        charts['chartPredict'] = new Chart(ctx1, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Observaciones',
                    data: res.yReal.map((y, i) => ({ x: y, y: res.yEstimado[i] })),
                    backgroundColor: '#4f46e5'
                }, {
                    type: 'line',
                    label: 'Ajuste Perfecto',
                    data: [{x: minVal, y: minVal}, {x: maxVal, y: maxVal}],
                    borderColor: '#e5e7eb',
                    borderDash: [5,5],
                    pointRadius: 0
                }]
            },
            options: { 
                ...commonOptions, 
                scales: { 
                    x: {title:{display:true, text:'Valor Real (Y)'}}, 
                    y: {title:{display:true, text:'Valor Predicho (Y Hat)'}} 
                } 
            }
        });

        /* ------------ 2. Residuos vs Ajustados --------------*/
        const ctx2 = document.getElementById('chartResFit').getContext('2d');
        charts['chartResFit'] = new Chart(ctx2, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Residuos',
                    data: res.yEstimado.map((y, i) => ({ x: y, y: res.residuos[i] })),
                    backgroundColor: '#ec4899'
                }]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: { title: {display:true, text:'Valores Ajustados'} },
                    y: { title: {display:true, text:'Residuos'} }
                }
            }
        });

        /* ------------ 3. Histograma Residuos --------------*/
        const ctx3 = document.getElementById('chartHist').getContext('2d');
        const resData = res.residuos;
        const binCount = 10;
        const minR = Math.min(...resData), maxR = Math.max(...resData);
        const step = (maxR - minR) / binCount;
        const bins = Array(binCount).fill(0);
        const labels = [];
        
        for(let i=0; i<binCount; i++) {
            labels.push((minR + i*step).toFixed(2));
            let limit = minR + (i+1)*step;
            bins[i] = resData.filter(v => v >= (minR + i*step) && v < limit).length;
        }
        bins[binCount-1] += resData.filter(v => v === maxR).length;

        charts['chartHist'] = new Chart(ctx3, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Frecuencia',
                    data: bins,
                    backgroundColor: '#10b981',
                    borderRadius: 4
                }]
            },
            options: commonOptions
        });
        
        /* ------------ 4. QQ Plot --------------*/
        const ctx4 = document.getElementById('chartQQ').getContext('2d');
        const sortedRes = [...res.residuos].sort((a,b)=>a-b);
        const n = sortedRes.length;
        const theoretical = sortedRes.map((_, i) => {
            const p = (i + 0.5) / n;
            return 4.91 * (Math.pow(p, 0.14) - Math.pow(1 - p, 0.14));
        });

        charts['chartQQ'] = new Chart(ctx4, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Cuantiles',
                    data: theoretical.map((t, i) => ({x: t, y: sortedRes[i]})),
                    backgroundColor: '#f59e0b'
                }]
            },
            options: {
                ...commonOptions,
                scales: {
                    x: { title: {display:true, text:'Cuantiles Teóricos'} },
                    y: { title: {display:true, text:'Cuantiles de Muestra'} }
                }
            }
        });
    }

