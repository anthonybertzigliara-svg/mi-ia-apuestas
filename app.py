if st.button("⚡ GENERAR PRONÓSTICO MAESTRO"):
        v = [[le.transform([t1])[0], le.transform([t2])[0], val1, valx, val2]]
        
        # Probabilidades detalladas
        probs = m_win.predict_proba(v)[0] # [Empate, Local, Visita]
        prob_empate, prob_local, prob_visita = probs[0], probs[1], probs[2]
        
        g, c, cards = m_goals.predict(v)[0], m_corn.predict(v)[0], m_cards.predict(v)[0]
        idx = m_win.predict(v)[0]
        ganador_final = t1 if idx == 1 else (t2 if idx == 2 else "Empate")

        # PANEL DE PREDICCIÓN PRINCIPAL
        st.markdown(f"""
            <div class="main-prediction">
                <h2 style="color: #00f2ff !important;">PICK RECOMENDADO</h2>
                <h1 style="margin: 20px 0;">{ganador_final.upper()}</h1>
                <p style="font-size: 1.5rem; font-weight: bold;">CONFIANZA DEL SISTEMA: {max(probs)*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # SECCIÓN ESTILO CASA DE APUESTAS
        st.markdown("### 📈 PROBABILIDADES REALES (IA)")
        p1, px, p2 = st.columns(3)
        
        # Función para detectar valor (Cuota * Probabilidad > 1)
        def check_value(cuota, prob):
            return "✅ VALOR" if (cuota * prob) > 1.05 else "❌ SIN VALOR"

        with p1:
            st.markdown(f"""<div class="result-card">
                <div class="metric-label">Gana {t1}</div>
                <div class="metric-value">{prob_local*100:.1f}%</div>
                <div style="color: #00ff88;">{check_value(val1, prob_local)}</div>
            </div>""", unsafe_allow_html=True)
        with px:
            st.markdown(f"""<div class="result-card">
                <div class="metric-label">Empate</div>
                <div class="metric-value">{prob_empate*100:.1f}%</div>
                <div style="color: #8899ac;">{check_value(valx, prob_empate)}</div>
            </div>""", unsafe_allow_html=True)
        with p2:
            st.markdown(f"""<div class="result-card">
                <div class="metric-label">Gana {t2}</div>
                <div class="metric-value">{prob_visita*100:.1f}%</div>
                <div style="color: #ff4b4b;">{check_value(val2, prob_visita)}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # GRID DE MÉTRICAS (Goles, Córners, Tarjetas)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class="result-card" style="border-left: 5px solid #00ff88;">
                <div class="metric-label">Goles Totales</div>
                <div class="metric-value">{g:.1f}</div>
                <div style="font-weight: bold;">{'OVER 2.5' if g > 2.5 else 'UNDER 2.5'}</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="result-card" style="border-left: 5px solid #ffcc00;">
                <div class="metric-label">Córners Est.</div>
                <div class="metric-value">{c:.1f}</div>
                <div style="font-weight: bold;">{'+9.5' if c > 9.5 else '-9.5'}</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="result-card" style="border-left: 5px solid #ff4b4b;">
                <div class="metric-label">Tarjetas Est.</div>
                <div class="metric-value">{cards:.1f}</div>
                <div style="font-weight: bold;">{'TENSO' if cards > 4.5 else 'LIMPIO'}</div>
            </div>""", unsafe_allow_html=True)
