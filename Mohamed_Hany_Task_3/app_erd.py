 """app.py — Simplified Uber Fare Prediction Dashboard for Ordinary Users"""
 import streamlit as st
 import utils
 import pandas as pd
 
 st.set_page_config(
     page_title="Uber NYC Ride Study",
     page_icon="🚕",
     layout="wide",
     initial_sidebar_state="expanded"
 )
 
 # ── Custom CSS ──────────────────────────────────────────────────────────────
 st.markdown("""
 <style>
 @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
 html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
 .stApp { background-color: #f8fafc; }
 
 [data-testid="stSidebar"] {
     background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%);
 }
 [data-testid="stSidebar"] * { color: #f8fafc !important; }
 [data-testid="stSidebar"] .stRadio label {
     display:block; padding:10px 14px; border-radius:8px;
     font-size:13.5px; font-weight:500; cursor:pointer;
     transition:background .2s;
 }
 [data-testid="stSidebar"] .stRadio label:hover { background:rgba(255,255,255,.07); }
 
 .card {
     background:#ffffff; border:1px solid #e2e8f0; border-radius:14px;
     padding:24px 28px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom:20px;
 }
 .metric-card {
     background:#ffffff; border:1px solid #e2e8f0; border-radius:12px;
     padding:18px 20px; box-shadow:0 2px 4px rgba(0,0,0,.04); height:100%;
 }
 .metric-label { font-size:11px; font-weight:600; color:#64748b;
     text-transform:uppercase; letter-spacing:.08em; }
 .metric-value { font-size:26px; font-weight:800; color:#0f172a; margin-top:4px; }
 .metric-desc { font-size:12px; color:#64748b; margin-top:2px; }
 
 .stTabs [data-baseweb="tab-list"] {
     background:#e2e8f0; padding:5px; border-radius:10px; gap:6px;
 }
 .stTabs [data-baseweb="tab"] {
     border-radius:7px; font-weight:600; color:#64748b;
     border:none !important; padding:10px 20px;
 }
 .stTabs [aria-selected="true"] {
     background:#ffffff !important; color:#2563eb !important;
     box-shadow:0 1px 3px rgba(0,0,0,0.1);
 }
 .insight-card {
     background:#ffffff; border:1px solid #e2e8f0; border-left:5px solid #2563eb;
     border-radius:12px; padding:22px; margin-bottom:16px;
     box-shadow:0 4px 6px -1px rgba(0,0,0,0.05);
 }
 .section-title { font-size:24px; font-weight:800; color:#0f172a; margin-bottom:18px; }
 .chart-note {
     background:#f0fdf4; border-left:4px solid #10b981;
     border-radius:6px; padding:12px 18px; margin-top:6px;
     font-size:14px; color:#1e293b; line-height:1.6;
     box-shadow:0 1px 2px rgba(0,0,0,0.02);
 }
 </style>
 """, unsafe_allow_html=True)
 
 
 # ── Helpers ──────────────────────────────────────────────────────────────────
 def render_metric(label, value, desc=""):
     desc_html = f'<div class="metric-desc">{desc}</div>' if desc else ""
     return f"""
     <div class="metric-card">
         <div class="metric-label">{label}</div>
         <div class="metric-value">{value}</div>
         {desc_html}
     </div>"""
 
 def note(text):
     st.markdown(f'<div class="chart-note">💡 <b>Quick Takeaway:</b> {text}</div>', unsafe_allow_html=True)
 
 
 # ── Sidebar Navigation ───────────────────────────────────────────────────────
 st.sidebar.markdown("""
 <div style="text-align:center;padding:20px 0 24px;border-bottom:1px solid rgba(255,255,255,.1);">
   <div style="font-size:36px;">🚕</div>
   <div style="font-size:19px;font-weight:800;color:#fff;margin-top:6px;">Uber NYC Ride Study</div>
   <div style="font-size:10px;color:#94a3b8;letter-spacing:1px;margin-top:4px;text-transform:uppercase;">
     Simple Dashboard for Users
   </div>
 </div>
 """, unsafe_allow_html=True)
 
 pages = [
     "🏠 Overview & Goals",
     "📊 Explore the Charts",
     "🛠️ How We Cleaned the Data",
     "💡 Interesting Findings",
 ]
 page = st.sidebar.radio("", pages, label_visibility="collapsed")
 
 st.sidebar.markdown("---")
 st.sidebar.markdown("""
 <div style="font-size:11.5px;color:#94a3b8;padding:0 4px;line-height:1.6;">
   <b style="color:#cbd5e1;">Study Focus:</b> Understanding what influences Uber pricing in New York City.<br><br>
   <b style="color:#cbd5e1;">Dataset Size:</b> 500,000 rides<br><br>
   <b style="color:#cbd5e1;">Developed for:</b> Cellula Technologies
 </div>
 """, unsafe_allow_html=True)
 
 
 # ── Load Data ─────────────────────────────────────────────────────────────────
 with st.spinner("⏳ Loading dataset..."):
     df_raw = utils.load_raw_preview()
     df, steps = utils.load_cleaned_data_with_steps()
 
 
 # ═══════════════════════════════════════════════════════════════════════════════
 # PAGE 1 — OVERVIEW & GOALS
 # ═══════════════════════════════════════════════════════════════════════════════
 if page == "🏠 Overview & Goals":
     st.markdown('<div class="section-title">🏠 Welcome & Study Overview</div>', unsafe_allow_html=True)
 
     st.markdown("""
     <div class="card">
       <h3 style="margin-top:0;color:#1e293b;">What is this study about?</h3>
       <p style="color:#475569;font-size:15px;line-height:1.8;margin:0;">
         We analyzed a massive dataset of <b>500,000 Uber rides</b> in New York City. 
         Our goal was simple: to figure out what actually determines the price of your ride, 
         how much demand varies throughout the day, and whether common factors like weather, 
         traffic, or the number of passengers actually affect how much you pay.
       </p>
     </div>
     """, unsafe_allow_html=True)
 
     # Key metrics row
     c1, c2, c3 = st.columns(3)
     with c1: st.markdown(render_metric("Total Rides Analyzed", "500,000", "Rides across NYC"), unsafe_allow_html=True)
     with c2: st.markdown(render_metric("Average Trip Fare", "$11.35", "Standard ride cost"), unsafe_allow_html=True)
     with c3: st.markdown(render_metric("Average Trip Distance", "2.12 km", "Typical ride length"), unsafe_allow_html=True)
 
     st.markdown("<br>", unsafe_allow_html=True)
     
     st.subheader("What questions did we answer?")
     st.markdown("""
     - 📏 **Distance:** Does a longer ride always cost more, and are there flat-rate routes?
     - 👥 **Passengers:** Do you pay more if you share the car with friends?
     - ⏰ **Timing:** What are the cheapest and most expensive hours to book a ride?
     - 🌧️ **Weather & Traffic:** Do rainy days or heavy gridlock make your trip more expensive?
     """)
 
     st.subheader("Raw Data Preview  (First 10 Rides)")
     st.dataframe(df_raw.head(10), use_container_width=True)
 
 
 # ═══════════════════════════════════════════════════════════════════════════════
 # PAGE 2 — EXPLORE THE CHARTS
 # ═══════════════════════════════════════════════════════════════════════════════
 elif page == "📊 Explore the Charts":
     st.markdown('<div class="section-title">📊 Explore the Charts</div>', unsafe_allow_html=True)
 
     # Control slider for scatter sample
     st.sidebar.markdown("---")
     st.sidebar.subheader("⚙️ Map & Chart Detail")
     scatter_n = st.sidebar.slider(
         "Detail Level (Sample size)", 5_000, 80_000, 30_000, 5_000,
         help="Higher values show more detail on scatter plots and maps."
     )
 
     tab1, tab2, tab3, tab4 = st.tabs([
         "📏 Distance & Prices",
         "⏰ Hours & Demand",
         "🗺️ Airport & NYC Map",
         "👥 Weather, Traffic & Passengers"
     ])
 
     # ── TAB 1 ─────────────────────────────────────────────────────────────
     with tab1:
         st.plotly_chart(utils.plot_distance_scatter(df, scatter_n), use_container_width=True)
         note("As you travel further, the price of your ride increases steadily. "
              "However, the horizontal lines of dots (specifically around $52) show that some rides "
              "have fixed flat prices regardless of distance, which usually happens on airport trips.")
 
         st.markdown("---")
         st.plotly_chart(utils.plot_correlation_matrix(df), use_container_width=True)
         note("This chart shows how different factors relate to price. "
              "Trip distance has by far the strongest connection to the fare. "
              "On the other hand, things like the number of passengers or the direction of travel "
              "have zero connection to how much you pay.")
 
         st.markdown("---")
         st.plotly_chart(utils.plot_fare_distribution(df), use_container_width=True)
         note("Most Uber rides in New York City are short and cheap, costing between $6.00 and $10.00. "
              "Expensive rides over $50 are rare and represent long trips or airport commutes.")
 
     # ── TAB 2 ─────────────────────────────────────────────────────────────
     with tab2:
         st.plotly_chart(utils.plot_hour_fare_trend(df), use_container_width=True)
         note("Rides are most expensive early in the morning, peaking at 5 AM with an average fare of $14.80. "
              "This is because of early morning airport runs and a shortage of available drivers. "
              "During the rest of the day (7 AM to 11 PM), prices remain very stable (around $11.00 to $11.80).")
 
         st.markdown("---")
         st.plotly_chart(utils.plot_ride_demand(df), use_container_width=True)
         note("The busiest time to book an Uber is in the evening between 6 PM and 10 PM, peaking at 7 PM. "
              "There is also a smaller peak at 8 AM as people head to work. "
              "The quietest time is late at night/early morning between 2 AM and 5 AM.")
 
     # ── TAB 3 ─────────────────────────────────────────────────────────────
     with tab3:
         st.plotly_chart(utils.plot_jfk_scatter(df, scatter_n), use_container_width=True)
         note("All rides starting or ending very close to JFK Airport show a flat price of $52.00. "
              "This matches NYC regulations which mandate flat fares between Manhattan and JFK, "
              "which Uber follows to remain competitive.")
 
         st.markdown("---")
         st.plotly_chart(utils.plot_geo_pickups(df, scatter_n), use_container_width=True)
         note("Each dot on this map is a ride pickup location. The bright colors show expensive fares. "
              "You can see that the most expensive rides start in Manhattan and end at the major airports.")
 
     # ── TAB 4 ─────────────────────────────────────────────────────────────
     with tab4:
         col1, col2 = st.columns(2)
         with col1:
             st.plotly_chart(utils.plot_passenger_count_rides(df), use_container_width=True)
         with col2:
             st.plotly_chart(utils.plot_passenger_count_box(df), use_container_width=True)
         note("Almost 80% of rides are taken by a single passenger. "
              "Importantly, the price of a ride is exactly the same whether you ride alone or "
              "with a group of up to 4 people. Sharing a ride is a great way to save money!")
 
         st.markdown("---")
         col3, col4 = st.columns(2)
         with col3:
             st.plotly_chart(utils.plot_car_condition_avg(df), use_container_width=True)
         with col4:
             st.plotly_chart(utils.plot_car_condition_box(df), use_container_width=True)
         note("The average price of a ride is $11.35 regardless of whether the car's condition "
              "is rated Bad, Good, Very Good, or Excellent. Uber does not charge you extra for a nicer car, "
              "nor does it discount rides for older cars.")
 
         st.markdown("---")
         st.plotly_chart(utils.plot_traffic_violin(df), use_container_width=True)
         note("The price distribution is identical across Flow, Dense, and Congested traffic. "
              "Because Uber uses upfront pricing computed at booking, you don't pay more "
              "if you get stuck in traffic during the trip.")
 
         st.markdown("---")
         st.plotly_chart(utils.plot_weather_distance(df), use_container_width=True)
         note("The distance of trips people take remains the same (~2.12 km) regardless of the weather. "
              "While bad weather (like storms or rain) makes more people request rides, "
              "it does not change the actual length of the trips they take.")
 
 
 # ═══════════════════════════════════════════════════════════════════════════════
 # PAGE 3 — HOW WE CLEANED THE DATA
 # ═══════════════════════════════════════════════════════════════════════════════
 elif page == "🛠️ How We Cleaned the Data":
     st.markdown('<div class="section-title">🛠️ Preparing and Cleaning the Data</div>', unsafe_allow_html=True)
 
     st.markdown("""
     <div class="card">
       <h3 style="margin-top:0;color:#1e293b;">Why clean the data?</h3>
       <p style="color:#475569;font-size:15px;line-height:1.8;margin:0;">
         Raw computer data from GPS devices often contains errors. For example, some rides had coordinate points 
         located in the middle of the Atlantic Ocean, negative fare amounts, or passenger counts of zero. 
         To make sure our charts are accurate, we cleaned the dataset step-by-step.
       </p>
     </div>
     """, unsafe_allow_html=True)
 
     c1, c2, c3 = st.columns(3)
     with c1: st.markdown(render_metric("Original Rides", "500,000", "Total raw dataset"), unsafe_allow_html=True)
     with c2: st.markdown(render_metric("Cleaned Rides", f"{len(df):,}", "Active, realistic rides"), unsafe_allow_html=True)
     with c3: st.markdown(render_metric("Rides Kept (%)", "87.5%", "High-quality data retained"), unsafe_allow_html=True)
 
     st.markdown("<br>", unsafe_allow_html=True)
     st.subheader("Key Cleaning Steps Performed")
     
     st.markdown("""
     1. ❌ **Removed Invalid Prices:** We removed rides that had negative prices or fares below the New York City minimum of $2.50. We also capped extremely expensive rides at $150.
     2. 🗺️ **Removed Bad GPS Locations:** We filtered out coordinate points that fell outside the New York City metropolitan area bounding box (which removes GPS glitches pointing to 0,0 longitude/latitude).
     3. 👥 **Valid Passenger Counts:** We removed rides listing 0 passengers, and restricted the analysis to standard passenger counts between 1 and 4.
     4. 📏 **Valid Distances:** We removed rides with 0 km distance (cancellations or waiting) and long-distance trips over 100 km.
     5. 🕵️ **Removed Names and Private Keys:** We removed identifiers like User Name and Driver Name to protect privacy.
     """)
 
 
 # ═══════════════════════════════════════════════════════════════════════════════
 # PAGE 4 — INTERESTING FINDINGS
 # ═══════════════════════════════════════════════════════════════════════════════
 elif page == "💡 Interesting Findings":
     st.markdown('<div class="section-title">💡 Key Takeaways for Uber Riders</div>', unsafe_allow_html=True)
 
     st.markdown("""
     <div class="insight-card">
         <h4 style="color:#2563eb;margin-top:0;font-size:16px;">🚗 1. Ride-Sharing Saves Money</h4>
         <p style="color:#475569;font-size:14.5px;line-height:1.6;margin:0;">
             Uber charges you for the trip's distance and duration, <b>not</b> the number of passengers. 
             A group of 4 passengers pays the exact same fare as a solo rider. Sharing a ride with friends 
             is the most effective way to cut your travel costs.
         </p>
     </div>
     
     <div class="insight-card">
         <h4 style="color:#2563eb;margin-top:0;font-size:16px;">⏰ 2. Avoid the 5 AM Surge</h4>
         <p style="color:#475569;font-size:14.5px;line-height:1.6;margin:0;">
             The average price of a ride peaks at 5 AM ($14.80), which is much higher than the rest of the day. 
             This is caused by early travelers heading to the airports and a shortage of active drivers on the road. 
             If you need an early morning ride, try booking slightly before or after this peak.
         </p>
     </div>
     
     <div class="insight-card">
         <h4 style="color:#2563eb;margin-top:0;font-size:16px;">✈️ 3. JFK Airport Flat Fares</h4>
         <p style="color:#475569;font-size:14.5px;line-height:1.6;margin:0;">
             Rides to and from JFK Airport have a flat price of $52.00 due to local New York City taxi laws. 
             This pricing overrides the standard distance meter, providing price predictability for airport travelers.
         </p>
     </div>
 
     <div class="insight-card">
         <h4 style="color:#2563eb;margin-top:0;font-size:16px;">🚦 4. Traffic and Car Ratings Do Not Change the Price</h4>
         <p style="color:#475569;font-size:14.5px;line-height:1.6;margin:0;">
             Because Uber calculates fares upfront before you accept the ride, getting stuck in traffic or riding in 
             a car with an older rating will not increase your price. The price you see when booking is the price you pay.
         </p>
     </div>
     
     <div class="insight-card">
         <h4 style="color:#2563eb;margin-top:0;font-size:16px;">🌧️ 5. Bad Weather Increases Wait Times, Not Distance</h4>
         <p style="color:#475569;font-size:14.5px;line-height:1.6;margin:0;">
             Rain and storms don't change how far people travel. While bad weather makes more people request rides 
             (which can trigger surge pricing due to high demand), the actual lengths of the trips remain short NYC runs.
         </p>
     </div>
     """, unsafe_allow_html=True)
 
 # ── Footer ────────────────────────────────────────────────────────────────────
 st.markdown("""
 <div style="text-align:center;color:#94a3b8;font-size:12px;
      margin-top:60px;padding-top:20px;border-top:1px solid #e2e8f0;">
   Uber NYC Ride Study • Developed for Cellula Technologies
 </div>
 """, unsafe_allow_html=True)
 