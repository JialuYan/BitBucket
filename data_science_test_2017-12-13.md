

```python
# import library
import pandas as pd
import numpy as np
from bokeh.io import output_notebook
import zipcode
from bokeh.io import show, output_file
from bokeh.models import HoverTool
from bokeh.plotting import figure
from bokeh.models import FactorRange
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
%matplotlib inline
output_notebook()
```



    <div class="bk-root">
        <a href="http://bokeh.pydata.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
        <span id="98f2b8ed-c37d-46e8-bd6d-efcf70fe651b">Loading BokehJS ...</span>
    </div>





```python
# set variables
```


```python
# define functions
def zipcodeMapper(code):
    code = str(int(code)).zfill(5)
    myzip = zipcode.isequal(code)
    try:
        return myzip.state
    except:
        return 'error'
```


```python
# calculate variables
```


```python
# readin data

zip_df  = pd.read_csv('free-zipcode-database.csv')
zip_df = zip_df[['Zipcode','State']].drop_duplicates()
zip_df['Zip'] = zip_df['Zipcode'].apply(lambda x:str(int(x)).zfill(5))
zip_df = zip_df.drop('Zipcode',axis=1)

dat = pd.read_csv('re_data.csv',index_col=0)
```

    /Users/pengdong/anaconda/lib/python3.6/site-packages/IPython/core/interactiveshell.py:2717: DtypeWarning: Columns (11) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
# EDA334
# check null values
print(len(dat))
dat.isnull().sum()
dat = dat.dropna()
print(dat.dtypes)
```

    967





    Maturity Date               15
    Loan Amount                  0
    Zip                         12
    Property Value               0
    Year Built                  23
    Net Operating Income         0
    Effective Gross Income       0
    Total Operating Expenses     0
    Maintenance Expense          0
    Parking Expense              0
    Taxes Expense                0
    Insurance Expense            0
    Utilities Expense            0
    Payroll Expense              0
    dtype: int64



    Maturity Date                object
    Loan Amount                 float64
    Zip                         float64
    Property Value               object
    Year Built                   object
    Net Operating Income        float64
    Effective Gross Income      float64
    Total Operating Expenses    float64
    Maintenance Expense         float64
    Parking Expense             float64
    Taxes Expense               float64
    Insurance Expense           float64
    Utilities Expense           float64
    Payroll Expense             float64
    dtype: object



```python
dat['Zip'] = dat['Zip'].apply(lambda x:str(int(x)).zfill(5))
dat = pd.merge(dat,zip_df,on = 'Zip',how = 'left')
dat['Property Value'] = dat['Property Value'].apply(float)
# what the fk
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-22-77b2f9e87d60> in <module>()
          1 dat['Zip'] = dat['Zip'].apply(lambda x:str(int(x)).zfill(5))
          2 dat = pd.merge(dat,zip_df,on = 'Zip',how = 'left')
    ----> 3 dat['Property Value'] = dat['Property Value'].apply(float)
          4 # what the fk


    /Users/pengdong/anaconda/lib/python3.6/site-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, **kwds)
       2292             else:
       2293                 values = self.asobject
    -> 2294                 mapped = lib.map_infer(values, f, convert=convert_dtype)
       2295 
       2296         if len(mapped) and isinstance(mapped[0], Series):


    pandas/src/inference.pyx in pandas.lib.map_infer (pandas/lib.c:66124)()


    ValueError: could not convert string to float: 'Error'



```python
dat = dat[dat['Property Value'] != 'Error']
dat['Property Value'] = dat['Property Value'].apply(float)
```


```python
dat.dtypes
```




    Maturity Date                object
    Loan Amount                 float64
    Zip                          object
    Property Value              float64
    Year Built                   object
    Net Operating Income        float64
    Effective Gross Income      float64
    Total Operating Expenses    float64
    Maintenance Expense         float64
    Parking Expense             float64
    Taxes Expense               float64
    Insurance Expense           float64
    Utilities Expense           float64
    Payroll Expense             float64
    State                        object
    dtype: object




```python
# 2 - a
dat_2a = dat.groupby('State',as_index=False).agg({'Loan Amount':'mean'}).sort_values(by = 'Loan Amount',ascending = False)
print('the state with highest average loan amount is {}'.format(dat_2a.iloc[0,0]))
```

    the state with highest average loan amount is OH



```python
# 2 - b
dat_2b = dat.groupby('State',as_index=False).agg({'Taxes Expense':'sum','Property Value':'sum'})                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

dat_2b['ratio'] = dat_2b['Taxes Expense']/dat_2b['Property Value']
dat_2b = dat_2b.sort_values(by = 'ratio',ascending = False)
print('the state with highest taxes as a % of property value is {}'.format(dat_2b.iloc[0,0]))
```

    the state with highest taxes as a % of property value is RI



```python
# 2 - c
dat_2c_origin = dat.copy()
dat_2c_origin['ratio'] = dat_2c_origin['Maintenance Expense']/dat_2c_origin['Property Value']
dat_2c = dat_2c_origin.corr()['ratio'].apply(abs).sort_values()# only focus on abs of corr
print("""strongest predictor might be {}, but this is just a rough guess, 
because in this way, we can only identify linear relations between features""".format(dat_2c.index[-2]))
```

    strongest predictor might be Maintenance Expense, but this is just a rough guess, 
    because in this way, we can only identify linear relations between features



```python
# 2 - d
dat_2d_origin = dat.copy()
dat_2d_origin = dat_2d_origin[dat_2d_origin['Property Value'] != 0]
dat_2d_origin['ratio'] = dat_2d_origin['Loan Amount']/dat_2d_origin['Property Value']

dat_2d = dat_2d_origin.ratio.describe()
print("""
median:{}
max:{}
min:{}
variance:{}
""".format(dat_2d['50%'],dat_2d['max'],dat_2d['min'],dat_2d['std']**2))
```

    
    median:0.6637972465882658
    max:1.6483758018592456
    min:0.0
    variance:0.06959396062440731
    



```python
# 2 - e

```


```python
# 3 - a
dat_3_a = dat.groupby('State',as_index=False).agg({'Total Operating Expenses':'sum','Effective Gross Income':'sum'})
dat_3_a['Expense ratio'] = dat_3_a['Total Operating Expenses']/dat_3_a['Effective Gross Income']

dat_3_a = dat_3_a[['State','Expense ratio']].sort_values(by = 'Expense ratio',ascending = False)

x3a = list(dat_3_a.State)
y3a = list(dat_3_a['Expense ratio'])
x = np.arange(len(x3a))
p = figure(x_range = x3a,plot_height=250,plot_width = 750,title = 'Expense Ratio Bar Chart')
p.vbar(x = x3a,top = y3a,width = 0.7)
p.yaxis.axis_label = 'Expense Ratio'
p.xaxis.axis_label = 'State'
p.xaxis.major_label_orientation = 1
show(p)
```




<div style="display: table;"><div style="display: table-row;"><div style="display: table-cell;"><b title="bokeh.models.renderers.GlyphRenderer">GlyphRenderer</b>(</div><div style="display: table-cell;">id&nbsp;=&nbsp;'815938cf-1e28-4c39-ad08-081c078fb022', <span id="f4c3cb8f-cebd-43d2-9573-5693d4dd1f7e" style="cursor: pointer;">&hellip;)</span></div></div><div class="5e398b03-dec4-46fb-8c08-46788162020e" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">data_source&nbsp;=&nbsp;ColumnDataSource(id='194c430e-fa57-4821-ad3e-446a7ffabce7', ...),</div></div><div class="5e398b03-dec4-46fb-8c08-46788162020e" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">glyph&nbsp;=&nbsp;VBar(id='8d84e31f-5b1c-462c-ad7e-17d348f3c535', ...),</div></div><div class="5e398b03-dec4-46fb-8c08-46788162020e" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">hover_glyph&nbsp;=&nbsp;None,</div></div><div class="5e398b03-dec4-46fb-8c08-46788162020e" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_callbacks&nbsp;=&nbsp;{},</div></div><div class="5e398b03-dec4-46fb-8c08-46788162020e" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">level&nbsp;=&nbsp;'glyph',</div></div><div class="5e398b03-dec4-46fb-8c08-46788162020e" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">name&nbsp;=&nbsp;None,</div></div><div class="5e398b03-dec4-46fb-8c08-46788162020e" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">nonselection_glyph&nbsp;=&nbsp;VBar(id='02a4a13d-d868-465a-b07f-3a5d28e65418', ...),</div></div><div class="5e398b03-dec4-46fb-8c08-46788162020e" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">selection_glyph&nbsp;=&nbsp;None,</div></div><div class="5e398b03-dec4-46fb-8c08-46788162020e" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">tags&nbsp;=&nbsp;[],</div></div><div class="5e398b03-dec4-46fb-8c08-46788162020e" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">visible&nbsp;=&nbsp;True,</div></div><div class="5e398b03-dec4-46fb-8c08-46788162020e" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_range_name&nbsp;=&nbsp;'default',</div></div><div class="5e398b03-dec4-46fb-8c08-46788162020e" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_range_name&nbsp;=&nbsp;'default')</div></div></div>
<script>
(function() {
  var expanded = false;
  var ellipsis = document.getElementById("f4c3cb8f-cebd-43d2-9573-5693d4dd1f7e");
  ellipsis.addEventListener("click", function() {
    var rows = document.getElementsByClassName("5e398b03-dec4-46fb-8c08-46788162020e");
    for (var i = 0; i < rows.length; i++) {
      var el = rows[i];
      el.style.display = expanded ? "none" : "table-row";
    }
    ellipsis.innerHTML = expanded ? "&hellip;)" : "&lsaquo;&lsaquo;&lsaquo;";
    expanded = !expanded;
  });
})();
</script>







    <div class="bk-root">
        <div class="bk-plotdiv" id="664f3660-00a0-4435-ac73-2e9e363d5e49"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        document.getElementById("664f3660-00a0-4435-ac73-2e9e363d5e49").textContent = "BokehJS successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("664f3660-00a0-4435-ac73-2e9e363d5e49");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid '664f3660-00a0-4435-ac73-2e9e363d5e49' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"84d8f140-de53-4933-8aad-b9d551e6d40c":{"roots":{"references":[{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"ed50cdc3-3667-40e3-872b-51a73ed8490e","type":"PanTool"},{"id":"f4e49b2c-2e3b-42d9-a541-7f4918902566","type":"WheelZoomTool"},{"id":"9c871e30-f0af-45ed-9f00-2c842728ef7e","type":"BoxZoomTool"},{"id":"5af4e7b1-ca48-45a0-a2b1-679c4f6c63b1","type":"SaveTool"},{"id":"5cf321df-0997-4949-88be-d70bc1daebc5","type":"ResetTool"},{"id":"72912e2f-d358-40bd-80ad-058b28703191","type":"HelpTool"}]},"id":"f9fbf40a-a560-4038-b69a-ff8cf149ba49","type":"Toolbar"},{"attributes":{"plot":{"id":"abd2cc79-be84-46d3-9456-8911000294e5","subtype":"Figure","type":"Plot"},"ticker":{"id":"3c9e890f-5fba-4a03-afea-af78dfed47d6","type":"CategoricalTicker"}},"id":"cc7d24dd-62bf-4f5e-81ab-5a3a6fb9aadc","type":"Grid"},{"attributes":{"axis_label":"Expense Ratio","formatter":{"id":"ed24d3ce-17e7-4e60-84d8-d2363bc130d0","type":"BasicTickFormatter"},"plot":{"id":"abd2cc79-be84-46d3-9456-8911000294e5","subtype":"Figure","type":"Plot"},"ticker":{"id":"8cbfc471-b97d-4145-9bec-9e3a0307fbae","type":"BasicTicker"}},"id":"f49f5bd4-7c41-4d43-a3e4-b1d8b2d8d52d","type":"LinearAxis"},{"attributes":{},"id":"8cbfc471-b97d-4145-9bec-9e3a0307fbae","type":"BasicTicker"},{"attributes":{"dimension":1,"plot":{"id":"abd2cc79-be84-46d3-9456-8911000294e5","subtype":"Figure","type":"Plot"},"ticker":{"id":"8cbfc471-b97d-4145-9bec-9e3a0307fbae","type":"BasicTicker"}},"id":"d33cd8bc-261e-4ebc-9955-1a54106f6e2d","type":"Grid"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"8fb4b7f1-47ef-4aec-a2e0-33a391ae6fd0","type":"BoxAnnotation"},{"attributes":{"plot":{"id":"abd2cc79-be84-46d3-9456-8911000294e5","subtype":"Figure","type":"Plot"}},"id":"ed50cdc3-3667-40e3-872b-51a73ed8490e","type":"PanTool"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"top":{"field":"top"},"width":{"value":0.7},"x":{"field":"x"}},"id":"02a4a13d-d868-465a-b07f-3a5d28e65418","type":"VBar"},{"attributes":{"plot":{"id":"abd2cc79-be84-46d3-9456-8911000294e5","subtype":"Figure","type":"Plot"}},"id":"f4e49b2c-2e3b-42d9-a541-7f4918902566","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"8fb4b7f1-47ef-4aec-a2e0-33a391ae6fd0","type":"BoxAnnotation"},"plot":{"id":"abd2cc79-be84-46d3-9456-8911000294e5","subtype":"Figure","type":"Plot"}},"id":"9c871e30-f0af-45ed-9f00-2c842728ef7e","type":"BoxZoomTool"},{"attributes":{"plot":{"id":"abd2cc79-be84-46d3-9456-8911000294e5","subtype":"Figure","type":"Plot"}},"id":"5af4e7b1-ca48-45a0-a2b1-679c4f6c63b1","type":"SaveTool"},{"attributes":{"plot":{"id":"abd2cc79-be84-46d3-9456-8911000294e5","subtype":"Figure","type":"Plot"}},"id":"5cf321df-0997-4949-88be-d70bc1daebc5","type":"ResetTool"},{"attributes":{},"id":"ed24d3ce-17e7-4e60-84d8-d2363bc130d0","type":"BasicTickFormatter"},{"attributes":{"plot":{"id":"abd2cc79-be84-46d3-9456-8911000294e5","subtype":"Figure","type":"Plot"}},"id":"72912e2f-d358-40bd-80ad-058b28703191","type":"HelpTool"},{"attributes":{},"id":"bdf6abb5-9619-4377-a198-60317362ef23","type":"CategoricalTickFormatter"},{"attributes":{},"id":"3c9e890f-5fba-4a03-afea-af78dfed47d6","type":"CategoricalTicker"},{"attributes":{"callback":null,"factors":["OK","MS","WI","SC","NM","AL","NV","MI","MO","OH","MN","TX","MD","KY","LA","KS","FL","GA","IN","TN","CT","NC","DC","AR","AZ","RI","VA","OR","IL","NJ","CA","CO","NH","DE","NE","NY","MA","UT","PA","WA"]},"id":"c88dada7-307c-4771-83d7-6baf49b841df","type":"FactorRange"},{"attributes":{"axis_label":"State","formatter":{"id":"bdf6abb5-9619-4377-a198-60317362ef23","type":"CategoricalTickFormatter"},"major_label_orientation":1,"plot":{"id":"abd2cc79-be84-46d3-9456-8911000294e5","subtype":"Figure","type":"Plot"},"ticker":{"id":"3c9e890f-5fba-4a03-afea-af78dfed47d6","type":"CategoricalTicker"}},"id":"25c88a5d-ed8b-44e8-8106-50152e4ad5b4","type":"CategoricalAxis"},{"attributes":{"callback":null},"id":"a5724d88-2286-405d-b550-2755a4f75964","type":"DataRange1d"},{"attributes":{"data_source":{"id":"194c430e-fa57-4821-ad3e-446a7ffabce7","type":"ColumnDataSource"},"glyph":{"id":"8d84e31f-5b1c-462c-ad7e-17d348f3c535","type":"VBar"},"hover_glyph":null,"nonselection_glyph":{"id":"02a4a13d-d868-465a-b07f-3a5d28e65418","type":"VBar"},"selection_glyph":null},"id":"815938cf-1e28-4c39-ad08-081c078fb022","type":"GlyphRenderer"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"top":{"field":"top"},"width":{"value":0.7},"x":{"field":"x"}},"id":"8d84e31f-5b1c-462c-ad7e-17d348f3c535","type":"VBar"},{"attributes":{"callback":null,"column_names":["x","top"],"data":{"top":[0.6014378015581192,0.5542376968202142,0.5528323657167938,0.546119718176193,0.5450432557543766,0.5317617116249033,0.5296170620465581,0.5212599289556561,0.5178642336511871,0.5138861050027256,0.5065515971786719,0.5043084054409648,0.5012792172445816,0.49653342878138423,0.49029638695302247,0.47638710587052135,0.4667032055619765,0.46581372954586275,0.4628618083165404,0.458184412110911,0.457764600190896,0.4575156116426886,0.4514970394652855,0.4485213173916523,0.4436497117805156,0.4289278382299122,0.4255283077103521,0.4212533671982747,0.41229107788572306,0.4098119683240397,0.40902344026628845,0.4064479542939314,0.4050319222664856,0.38023394175655034,0.3740116072007246,0.366242964735565,0.3627848132286779,0.35354973672522033,0.34146783836260486,0.33274383256156886],"x":["OK","MS","WI","SC","NM","AL","NV","MI","MO","OH","MN","TX","MD","KY","LA","KS","FL","GA","IN","TN","CT","NC","DC","AR","AZ","RI","VA","OR","IL","NJ","CA","CO","NH","DE","NE","NY","MA","UT","PA","WA"]}},"id":"194c430e-fa57-4821-ad3e-446a7ffabce7","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"25c88a5d-ed8b-44e8-8106-50152e4ad5b4","type":"CategoricalAxis"}],"left":[{"id":"f49f5bd4-7c41-4d43-a3e4-b1d8b2d8d52d","type":"LinearAxis"}],"plot_height":250,"plot_width":750,"renderers":[{"id":"25c88a5d-ed8b-44e8-8106-50152e4ad5b4","type":"CategoricalAxis"},{"id":"cc7d24dd-62bf-4f5e-81ab-5a3a6fb9aadc","type":"Grid"},{"id":"f49f5bd4-7c41-4d43-a3e4-b1d8b2d8d52d","type":"LinearAxis"},{"id":"d33cd8bc-261e-4ebc-9955-1a54106f6e2d","type":"Grid"},{"id":"8fb4b7f1-47ef-4aec-a2e0-33a391ae6fd0","type":"BoxAnnotation"},{"id":"815938cf-1e28-4c39-ad08-081c078fb022","type":"GlyphRenderer"}],"title":{"id":"5402544b-df59-4f86-99ff-7557a7767928","type":"Title"},"tool_events":{"id":"802d7835-cf62-4a32-b023-610f6277052a","type":"ToolEvents"},"toolbar":{"id":"f9fbf40a-a560-4038-b69a-ff8cf149ba49","type":"Toolbar"},"x_range":{"id":"c88dada7-307c-4771-83d7-6baf49b841df","type":"FactorRange"},"y_range":{"id":"a5724d88-2286-405d-b550-2755a4f75964","type":"DataRange1d"}},"id":"abd2cc79-be84-46d3-9456-8911000294e5","subtype":"Figure","type":"Plot"},{"attributes":{"plot":null,"text":"Expense Ratio Bar Chart"},"id":"5402544b-df59-4f86-99ff-7557a7767928","type":"Title"},{"attributes":{},"id":"802d7835-cf62-4a32-b023-610f6277052a","type":"ToolEvents"}],"root_ids":["abd2cc79-be84-46d3-9456-8911000294e5"]},"title":"Bokeh Application","version":"0.12.4"}};
            var render_items = [{"docid":"84d8f140-de53-4933-8aad-b9d551e6d40c","elementid":"664f3660-00a0-4435-ac73-2e9e363d5e49","modelid":"abd2cc79-be84-46d3-9456-8911000294e5"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("664f3660-00a0-4435-ac73-2e9e363d5e49")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>



```python
# 3 - b
def errorSignal(x):
    try:
        tmp = int(x)
        return 'healthy'
    except:
        return 'error'
dat['built year error signal'] = dat['Year Built'].apply(errorSignal)
dat = dat[dat['built year error signal'] == 'healthy']
dat['Property Age'] = 2017 - dat['Year Built'].apply(int)

x = dat['Property Age']
y = dat['Total Operating Expenses']
p = figure(plot_height = 500,plot_width = 500,title = 'Relation Between Property Age and Expenses')
p.scatter(x,y)
p.yaxis.axis_label = 'Total Operating Expenses'
p.xaxis.axis_label = 'Property Age'
show(p)
```




<div style="display: table;"><div style="display: table-row;"><div style="display: table-cell;"><b title="bokeh.models.renderers.GlyphRenderer">GlyphRenderer</b>(</div><div style="display: table-cell;">id&nbsp;=&nbsp;'c4549120-5a98-408d-8ede-7901f990bb66', <span id="c0827bb6-9fdd-48ea-8837-a3bd6256a7fb" style="cursor: pointer;">&hellip;)</span></div></div><div class="89284bfa-55bf-4a3e-ac9d-14666065cde5" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">data_source&nbsp;=&nbsp;ColumnDataSource(id='81dd0a3f-8906-41c7-8dcb-e4801cad72ed', ...),</div></div><div class="89284bfa-55bf-4a3e-ac9d-14666065cde5" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">glyph&nbsp;=&nbsp;Circle(id='31fa1131-d5bc-4679-993d-fe3a1ee0733f', ...),</div></div><div class="89284bfa-55bf-4a3e-ac9d-14666065cde5" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">hover_glyph&nbsp;=&nbsp;None,</div></div><div class="89284bfa-55bf-4a3e-ac9d-14666065cde5" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_callbacks&nbsp;=&nbsp;{},</div></div><div class="89284bfa-55bf-4a3e-ac9d-14666065cde5" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">level&nbsp;=&nbsp;'glyph',</div></div><div class="89284bfa-55bf-4a3e-ac9d-14666065cde5" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">name&nbsp;=&nbsp;None,</div></div><div class="89284bfa-55bf-4a3e-ac9d-14666065cde5" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">nonselection_glyph&nbsp;=&nbsp;Circle(id='decb6576-2eee-45ce-bd69-69241037838e', ...),</div></div><div class="89284bfa-55bf-4a3e-ac9d-14666065cde5" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">selection_glyph&nbsp;=&nbsp;None,</div></div><div class="89284bfa-55bf-4a3e-ac9d-14666065cde5" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">tags&nbsp;=&nbsp;[],</div></div><div class="89284bfa-55bf-4a3e-ac9d-14666065cde5" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">visible&nbsp;=&nbsp;True,</div></div><div class="89284bfa-55bf-4a3e-ac9d-14666065cde5" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_range_name&nbsp;=&nbsp;'default',</div></div><div class="89284bfa-55bf-4a3e-ac9d-14666065cde5" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_range_name&nbsp;=&nbsp;'default')</div></div></div>
<script>
(function() {
  var expanded = false;
  var ellipsis = document.getElementById("c0827bb6-9fdd-48ea-8837-a3bd6256a7fb");
  ellipsis.addEventListener("click", function() {
    var rows = document.getElementsByClassName("89284bfa-55bf-4a3e-ac9d-14666065cde5");
    for (var i = 0; i < rows.length; i++) {
      var el = rows[i];
      el.style.display = expanded ? "none" : "table-row";
    }
    ellipsis.innerHTML = expanded ? "&hellip;)" : "&lsaquo;&lsaquo;&lsaquo;";
    expanded = !expanded;
  });
})();
</script>







    <div class="bk-root">
        <div class="bk-plotdiv" id="097c86b1-301c-48c1-9caf-a99091716ee0"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        document.getElementById("097c86b1-301c-48c1-9caf-a99091716ee0").textContent = "BokehJS successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("097c86b1-301c-48c1-9caf-a99091716ee0");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid '097c86b1-301c-48c1-9caf-a99091716ee0' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"1c993fd8-6d4e-4cd7-8d93-2f835a2f4dc5":{"roots":{"references":[{"attributes":{"callback":null},"id":"87881f03-fda4-48f6-90a5-a5a523a50964","type":"DataRange1d"},{"attributes":{"axis_label":"Property Age","formatter":{"id":"19308f1a-7b63-430a-9f84-bdfe3b4861b6","type":"BasicTickFormatter"},"plot":{"id":"c339948c-aef8-4fe0-969b-768013381ccb","subtype":"Figure","type":"Plot"},"ticker":{"id":"6186261d-cd95-4ed1-8a92-3b89ddb64573","type":"BasicTicker"}},"id":"852b3b7c-3806-48de-96d3-8240e7b0e65e","type":"LinearAxis"},{"attributes":{"fill_color":{"value":"#1f77b4"},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"31fa1131-d5bc-4679-993d-fe3a1ee0733f","type":"Circle"},{"attributes":{"overlay":{"id":"883f9e57-e006-4fc6-964c-ac9592ef4bc2","type":"BoxAnnotation"},"plot":{"id":"c339948c-aef8-4fe0-969b-768013381ccb","subtype":"Figure","type":"Plot"}},"id":"a04694af-6a35-4e56-b0ef-c55600078929","type":"BoxZoomTool"},{"attributes":{},"id":"f3b3373f-ecb7-4193-a32d-09e2d947bde9","type":"BasicTicker"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"75d3364c-b36b-4e9b-b9c2-58c57f76a2a0","type":"PanTool"},{"id":"597a31ff-b1de-4abf-a699-c12983ddf7c1","type":"WheelZoomTool"},{"id":"a04694af-6a35-4e56-b0ef-c55600078929","type":"BoxZoomTool"},{"id":"aa6fb428-cf69-47e4-b796-dba4917df0a1","type":"SaveTool"},{"id":"abbc59e0-5912-4736-86a4-97b9f21ef7e6","type":"ResetTool"},{"id":"ff5e7535-d159-410e-880e-ca7fa5aea788","type":"HelpTool"}]},"id":"4981664f-cce9-4fcc-86e6-482243f73f5f","type":"Toolbar"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"883f9e57-e006-4fc6-964c-ac9592ef4bc2","type":"BoxAnnotation"},{"attributes":{"plot":{"id":"c339948c-aef8-4fe0-969b-768013381ccb","subtype":"Figure","type":"Plot"}},"id":"597a31ff-b1de-4abf-a699-c12983ddf7c1","type":"WheelZoomTool"},{"attributes":{"plot":{"id":"c339948c-aef8-4fe0-969b-768013381ccb","subtype":"Figure","type":"Plot"},"ticker":{"id":"6186261d-cd95-4ed1-8a92-3b89ddb64573","type":"BasicTicker"}},"id":"1e44019a-4834-4b33-99d8-f9f6dce4a203","type":"Grid"},{"attributes":{},"id":"eda0e0ba-4ebe-48ba-8ba4-32067884849f","type":"ToolEvents"},{"attributes":{"plot":{"id":"c339948c-aef8-4fe0-969b-768013381ccb","subtype":"Figure","type":"Plot"}},"id":"abbc59e0-5912-4736-86a4-97b9f21ef7e6","type":"ResetTool"},{"attributes":{"plot":{"id":"c339948c-aef8-4fe0-969b-768013381ccb","subtype":"Figure","type":"Plot"}},"id":"75d3364c-b36b-4e9b-b9c2-58c57f76a2a0","type":"PanTool"},{"attributes":{"axis_label":"Total Operating Expenses","formatter":{"id":"5eaa66af-11d0-428d-9a18-d497035b5398","type":"BasicTickFormatter"},"plot":{"id":"c339948c-aef8-4fe0-969b-768013381ccb","subtype":"Figure","type":"Plot"},"ticker":{"id":"f3b3373f-ecb7-4193-a32d-09e2d947bde9","type":"BasicTicker"}},"id":"e1e00bb4-78c6-4139-9241-738f64a92eae","type":"LinearAxis"},{"attributes":{"plot":{"id":"c339948c-aef8-4fe0-969b-768013381ccb","subtype":"Figure","type":"Plot"}},"id":"ff5e7535-d159-410e-880e-ca7fa5aea788","type":"HelpTool"},{"attributes":{"plot":null,"text":"Relation Between Property Age and Expenses"},"id":"30a5e67f-046a-4b56-b77b-b37cfc68720e","type":"Title"},{"attributes":{"below":[{"id":"852b3b7c-3806-48de-96d3-8240e7b0e65e","type":"LinearAxis"}],"left":[{"id":"e1e00bb4-78c6-4139-9241-738f64a92eae","type":"LinearAxis"}],"plot_height":500,"plot_width":500,"renderers":[{"id":"852b3b7c-3806-48de-96d3-8240e7b0e65e","type":"LinearAxis"},{"id":"1e44019a-4834-4b33-99d8-f9f6dce4a203","type":"Grid"},{"id":"e1e00bb4-78c6-4139-9241-738f64a92eae","type":"LinearAxis"},{"id":"3c3ebb66-9846-4f9c-915c-99063ce28d0c","type":"Grid"},{"id":"883f9e57-e006-4fc6-964c-ac9592ef4bc2","type":"BoxAnnotation"},{"id":"c4549120-5a98-408d-8ede-7901f990bb66","type":"GlyphRenderer"}],"title":{"id":"30a5e67f-046a-4b56-b77b-b37cfc68720e","type":"Title"},"tool_events":{"id":"eda0e0ba-4ebe-48ba-8ba4-32067884849f","type":"ToolEvents"},"toolbar":{"id":"4981664f-cce9-4fcc-86e6-482243f73f5f","type":"Toolbar"},"x_range":{"id":"87881f03-fda4-48f6-90a5-a5a523a50964","type":"DataRange1d"},"y_range":{"id":"9662cea1-cef3-426f-b6b7-fd7857815abc","type":"DataRange1d"}},"id":"c339948c-aef8-4fe0-969b-768013381ccb","subtype":"Figure","type":"Plot"},{"attributes":{"plot":{"id":"c339948c-aef8-4fe0-969b-768013381ccb","subtype":"Figure","type":"Plot"}},"id":"aa6fb428-cf69-47e4-b796-dba4917df0a1","type":"SaveTool"},{"attributes":{},"id":"6186261d-cd95-4ed1-8a92-3b89ddb64573","type":"BasicTicker"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"decb6576-2eee-45ce-bd69-69241037838e","type":"Circle"},{"attributes":{"callback":null},"id":"9662cea1-cef3-426f-b6b7-fd7857815abc","type":"DataRange1d"},{"attributes":{},"id":"5eaa66af-11d0-428d-9a18-d497035b5398","type":"BasicTickFormatter"},{"attributes":{},"id":"19308f1a-7b63-430a-9f84-bdfe3b4861b6","type":"BasicTickFormatter"},{"attributes":{"dimension":1,"plot":{"id":"c339948c-aef8-4fe0-969b-768013381ccb","subtype":"Figure","type":"Plot"},"ticker":{"id":"f3b3373f-ecb7-4193-a32d-09e2d947bde9","type":"BasicTicker"}},"id":"3c3ebb66-9846-4f9c-915c-99063ce28d0c","type":"Grid"},{"attributes":{"data_source":{"id":"81dd0a3f-8906-41c7-8dcb-e4801cad72ed","type":"ColumnDataSource"},"glyph":{"id":"31fa1131-d5bc-4679-993d-fe3a1ee0733f","type":"Circle"},"hover_glyph":null,"nonselection_glyph":{"id":"decb6576-2eee-45ce-bd69-69241037838e","type":"Circle"},"selection_glyph":null},"id":"c4549120-5a98-408d-8ede-7901f990bb66","type":"GlyphRenderer"},{"attributes":{"callback":null,"column_names":["x","y"],"data":{"x":[52,57,58,41,65,43,93,74,89,90,27,38,45,86,44,36,56,20,3,34,52,25,33,143,89,75,44,43,11,39,87,88,46,149,44,87,45,2,112,92,55,14,38,86,49,112,87,52,103,27,29,48,2,47,57,86,34,8,88,38,97,41,112,65,108,44,117,1,42,44,48,33,67,45,38,44,86,47,34,103,86,64,55,97,79,8,48,33,32,52,118,38,105,50,45,48,39,43,107,100,107,43,97,38,107,112,101,39,92,55,86,29,49,86,11,2,34,1,43,47,51,30,44,59,50,50,53,29,44,48,87,33,33,97,43,35,108,136,86,68,29,111,49,43,32,86,87,34,117,46,37,33,31,86,49,32,77,110,86,32,94,31,35,107,63,97,45,32,32,57,53,86,16,117,45,42,30,39,75,34,48,87,49,4,29,21,33,86,132,16,2,42,89,32,91,55,86,91,56,32,86,67,97,107,86,58,87,123,39,32,49,44,39,92,34,106,86,93,30,112,45,93,1,57,44,41,120,45,97,93,112,77,2,22,97,107,44,41,117,54,21,1,37,87,66,104,11,58,47,86,17,55,42,95,53,66,52,44,43,47,86,107,89,82,50,122,108,93,44,95,47,44,92,52,0,43,29,34,107,32,51,29,43,51,33,39,102,87,85,1,67,30,38,55,105,27,87,114,1,46,116,101,123,3,55,47,72,30,89,43,52,52,28,8,32,88,2,45,59,42,46,77,97,86,47,41,77,42,68,2,109,93,37,97,17,36,45,42,2,9,55,22,87,97,46,52,44,107,1,32,80,44,14,51,2,82,90,60,9,18,67,97,2,86,18,28,88,47,112,55,46,45,91,89,41,34,32,87,33,86,46,37,79,50,97,53,87,47,89,41,90,2,44,89,43,85,117,29,56,87,48,45,33,11,22,35,88,76,107,34,91,10,46,48,117,86,89,86,86,40,92,87,52,44,29,33,46,39,95,44,86,94,125,45,103,86,107,86,51,50,89,130,111,11,44,43,100,37,33,48,24,3,43,39,45,10,86,53,12,42,14,64,32,117,3,1,40,105,87,118,86,62,86,30,76,86,82,32,45,91,87,86,107,10,1,49,90,101,51,50,29,48,117,87,47,25,92,53,86,55,6,86,86,86,1,55,55,117,87,107,38,86,97,49,29,87,54,31,47,47,107,86,86,29,86,53,62,102,42,39,87,77,45,87,86,54,16,52,135,1,45,117,116,43,56,21,22,45,86,121,48,59,20,31,44,27,117,39,86,57,12,37,2,97,112,91,43,24,105,41,27,86,42,54,17,33,55,87,49,53,97,54,84,31,93,107,35,33,60,1,97,89,48,42,32,58,44,34,54,31,33,86,54,53,86,86,87,86,56,21,60,44,89,1,56,46,43,95,46,107,42,62,94,86,43,97,31,56,33,32,34,32,111,34,44,93,86,21,33,58,86,47,33,86,44,87,30,29,19,54,86,119,112,67,32,86,92,95,86,9,107,28,54,87,42,34,128,87,11,9,91,54,91,31,86,87,92,117,67,22,55,102,29,45,43,44,89,77,77,95,59,43,42,97,86,34,13,43,49,91,29,87,87,87,37,54,54,37,9,41,54,11,37,5,46,41,48,47,13,31,86,54,49,49,52,86,24,58,55,87,87,47,55,3,9,38,87,40,46,52,87,10,2,86,56,57,93,33,86,49,86,87,2,54,29,31,56,10,98,118,4,45,52,38,54,93,86,110,107,38,86,111,82,61,3,92,57,49,92,11,32,33,117,100,37,107,104,69,8,91,30,24,94,10,37,40,92,97,86,86,54,58,87,1,29,61,54,6,107,88,54,1,43,108,45,36,1,86,17,19,32,52,1,12,118,52,67,52,10,20,49,86,112,41,56,45,10,41,30,30,46,107,62,62,93,87,49,32,18,56,96,88,86,112,52,52,46,48,87,97,53,86,77,2,97,92,86,38,52,107,112,1,112,3,97,2,57,30,39,44,47,117,2,47,86,97,86,4,29,49,86,97,92,38,86,45,49,91,38,86,105,30,86,27,107,91,86,86,10,107,86,15,97,86],"y":{"__ndarray__":"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","dtype":"float64","shape":[914]}}},"id":"81dd0a3f-8906-41c7-8dcb-e4801cad72ed","type":"ColumnDataSource"}],"root_ids":["c339948c-aef8-4fe0-969b-768013381ccb"]},"title":"Bokeh Application","version":"0.12.4"}};
            var render_items = [{"docid":"1c993fd8-6d4e-4cd7-8d93-2f835a2f4dc5","elementid":"097c86b1-301c-48c1-9caf-a99091716ee0","modelid":"c339948c-aef8-4fe0-969b-768013381ccb"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("097c86b1-301c-48c1-9caf-a99091716ee0")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>



```python
dat['Maturity Date'] = dat['Maturity Date'] \
    .apply(lambda x:datetime.strptime(x,'%Y-%m-%d'))
dat['Fiscal Year'] = dat['Maturity Date'].dt.year
dat['Fiscal Quarter'] = dat['Maturity Date'].dt.quarter
```


```python
def appendFiscalQuarter(row):
    return str(row[0])+'Q'+str(row[1])
dat['Fiscal Year Quarter'] = dat[['Fiscal Year','Fiscal Quarter']].apply(appendFiscalQuarter,axis=1)
```


```python
dat_3_c = dat.groupby('Fiscal Year Quarter',as_index=False).agg({'Loan Amount':'sum'}).sort_values(by = 'Fiscal Year Quarter' )
x3c = dat_3_c['Fiscal Year Quarter']
y3c = dat_3_c['Loan Amount']
p = figure(x_range = FactorRange(*x3c),plot_height =250,plot_width = 1000,title = 'Stacked Loan Amount by Fiscal Quarter')
p.line(x = x3c,y = y3c)
p.xaxis.major_label_orientation = 1
show(p)
```




<div style="display: table;"><div style="display: table-row;"><div style="display: table-cell;"><b title="bokeh.models.renderers.GlyphRenderer">GlyphRenderer</b>(</div><div style="display: table-cell;">id&nbsp;=&nbsp;'55a119fb-e583-4897-a3d1-b53f526303e8', <span id="d43ada9b-7d50-41d4-82c2-11e02296a404" style="cursor: pointer;">&hellip;)</span></div></div><div class="7ff0c045-b97d-45bd-bd15-b771a46f4254" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">data_source&nbsp;=&nbsp;ColumnDataSource(id='aaa3e294-a518-4ed7-a235-446194141cff', ...),</div></div><div class="7ff0c045-b97d-45bd-bd15-b771a46f4254" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">glyph&nbsp;=&nbsp;Line(id='1bc08687-8b4f-4801-9c65-cb60ebe40924', ...),</div></div><div class="7ff0c045-b97d-45bd-bd15-b771a46f4254" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">hover_glyph&nbsp;=&nbsp;None,</div></div><div class="7ff0c045-b97d-45bd-bd15-b771a46f4254" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">js_callbacks&nbsp;=&nbsp;{},</div></div><div class="7ff0c045-b97d-45bd-bd15-b771a46f4254" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">level&nbsp;=&nbsp;'glyph',</div></div><div class="7ff0c045-b97d-45bd-bd15-b771a46f4254" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">name&nbsp;=&nbsp;None,</div></div><div class="7ff0c045-b97d-45bd-bd15-b771a46f4254" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">nonselection_glyph&nbsp;=&nbsp;Line(id='4e5068dc-3162-49a4-8946-59a0efb85147', ...),</div></div><div class="7ff0c045-b97d-45bd-bd15-b771a46f4254" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">selection_glyph&nbsp;=&nbsp;None,</div></div><div class="7ff0c045-b97d-45bd-bd15-b771a46f4254" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">tags&nbsp;=&nbsp;[],</div></div><div class="7ff0c045-b97d-45bd-bd15-b771a46f4254" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">visible&nbsp;=&nbsp;True,</div></div><div class="7ff0c045-b97d-45bd-bd15-b771a46f4254" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">x_range_name&nbsp;=&nbsp;'default',</div></div><div class="7ff0c045-b97d-45bd-bd15-b771a46f4254" style="display: none;"><div style="display: table-cell;"></div><div style="display: table-cell;">y_range_name&nbsp;=&nbsp;'default')</div></div></div>
<script>
(function() {
  var expanded = false;
  var ellipsis = document.getElementById("d43ada9b-7d50-41d4-82c2-11e02296a404");
  ellipsis.addEventListener("click", function() {
    var rows = document.getElementsByClassName("7ff0c045-b97d-45bd-bd15-b771a46f4254");
    for (var i = 0; i < rows.length; i++) {
      var el = rows[i];
      el.style.display = expanded ? "none" : "table-row";
    }
    ellipsis.innerHTML = expanded ? "&hellip;)" : "&lsaquo;&lsaquo;&lsaquo;";
    expanded = !expanded;
  });
})();
</script>







    <div class="bk-root">
        <div class="bk-plotdiv" id="cd8723a0-6912-4708-9629-bc4e0b658471"></div>
    </div>
<script type="text/javascript">
  
  (function(global) {
    function now() {
      return new Date();
    }
  
    var force = false;
  
    if (typeof (window._bokeh_onload_callbacks) === "undefined" || force === true) {
      window._bokeh_onload_callbacks = [];
      window._bokeh_is_loading = undefined;
    }
  
  
    
    if (typeof (window._bokeh_timeout) === "undefined" || force === true) {
      window._bokeh_timeout = Date.now() + 0;
      window._bokeh_failed_load = false;
    }
  
    var NB_LOAD_WARNING = {'data': {'text/html':
       "<div style='background-color: #fdd'>\n"+
       "<p>\n"+
       "BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \n"+
       "may be due to a slow or bad network connection. Possible fixes:\n"+
       "</p>\n"+
       "<ul>\n"+
       "<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\n"+
       "<li>use INLINE resources instead, as so:</li>\n"+
       "</ul>\n"+
       "<code>\n"+
       "from bokeh.resources import INLINE\n"+
       "output_notebook(resources=INLINE)\n"+
       "</code>\n"+
       "</div>"}};
  
    function display_loaded() {
      if (window.Bokeh !== undefined) {
        document.getElementById("cd8723a0-6912-4708-9629-bc4e0b658471").textContent = "BokehJS successfully loaded.";
      } else if (Date.now() < window._bokeh_timeout) {
        setTimeout(display_loaded, 100)
      }
    }
  
    function run_callbacks() {
      window._bokeh_onload_callbacks.forEach(function(callback) { callback() });
      delete window._bokeh_onload_callbacks
      console.info("Bokeh: all callbacks have finished");
    }
  
    function load_libs(js_urls, callback) {
      window._bokeh_onload_callbacks.push(callback);
      if (window._bokeh_is_loading > 0) {
        console.log("Bokeh: BokehJS is being loaded, scheduling callback at", now());
        return null;
      }
      if (js_urls == null || js_urls.length === 0) {
        run_callbacks();
        return null;
      }
      console.log("Bokeh: BokehJS not loaded, scheduling load and callback at", now());
      window._bokeh_is_loading = js_urls.length;
      for (var i = 0; i < js_urls.length; i++) {
        var url = js_urls[i];
        var s = document.createElement('script');
        s.src = url;
        s.async = false;
        s.onreadystatechange = s.onload = function() {
          window._bokeh_is_loading--;
          if (window._bokeh_is_loading === 0) {
            console.log("Bokeh: all BokehJS libraries loaded");
            run_callbacks()
          }
        };
        s.onerror = function() {
          console.warn("failed to load library " + url);
        };
        console.log("Bokeh: injecting script tag for BokehJS library: ", url);
        document.getElementsByTagName("head")[0].appendChild(s);
      }
    };var element = document.getElementById("cd8723a0-6912-4708-9629-bc4e0b658471");
    if (element == null) {
      console.log("Bokeh: ERROR: autoload.js configured with elementid 'cd8723a0-6912-4708-9629-bc4e0b658471' but no matching script tag was found. ")
      return false;
    }
  
    var js_urls = [];
  
    var inline_js = [
      function(Bokeh) {
        (function() {
          var fn = function() {
            var docs_json = {"32a9e915-6572-45c7-b10a-a6574c6a385e":{"roots":{"references":[{"attributes":{},"id":"cb6479f0-59fe-435b-af53-d15c6283f435","type":"CategoricalTicker"},{"attributes":{"plot":{"id":"cf85504c-4c14-455c-a34f-eb6f13e68186","subtype":"Figure","type":"Plot"}},"id":"b80cdef7-983d-4b1e-b0ce-9a523bdd58ca","type":"PanTool"},{"attributes":{"plot":{"id":"cf85504c-4c14-455c-a34f-eb6f13e68186","subtype":"Figure","type":"Plot"}},"id":"8be8a7db-636b-4486-979d-39021a61e339","type":"WheelZoomTool"},{"attributes":{"overlay":{"id":"52a316fb-2dd5-4e7e-9865-365b861a5692","type":"BoxAnnotation"},"plot":{"id":"cf85504c-4c14-455c-a34f-eb6f13e68186","subtype":"Figure","type":"Plot"}},"id":"1e5f757a-4bb8-407c-8960-0f9ddf9f9550","type":"BoxZoomTool"},{"attributes":{"plot":{"id":"cf85504c-4c14-455c-a34f-eb6f13e68186","subtype":"Figure","type":"Plot"}},"id":"b596be80-8911-4d78-8d0e-9c421b66dbd8","type":"ResetTool"},{"attributes":{"plot":{"id":"cf85504c-4c14-455c-a34f-eb6f13e68186","subtype":"Figure","type":"Plot"}},"id":"330d3cc0-1199-4287-9806-6f2ade567297","type":"SaveTool"},{"attributes":{"plot":{"id":"cf85504c-4c14-455c-a34f-eb6f13e68186","subtype":"Figure","type":"Plot"}},"id":"f791276f-e7ff-4a4b-83a1-7e58ff5d16e0","type":"HelpTool"},{"attributes":{"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"1bc08687-8b4f-4801-9c65-cb60ebe40924","type":"Line"},{"attributes":{},"id":"7b58627c-1b8d-49dd-87fb-efaf89354f53","type":"BasicTicker"},{"attributes":{"dimension":1,"plot":{"id":"cf85504c-4c14-455c-a34f-eb6f13e68186","subtype":"Figure","type":"Plot"},"ticker":{"id":"7b58627c-1b8d-49dd-87fb-efaf89354f53","type":"BasicTicker"}},"id":"d64e6dbd-25a3-4acd-8ecf-26e1778372a2","type":"Grid"},{"attributes":{"data_source":{"id":"aaa3e294-a518-4ed7-a235-446194141cff","type":"ColumnDataSource"},"glyph":{"id":"1bc08687-8b4f-4801-9c65-cb60ebe40924","type":"Line"},"hover_glyph":null,"nonselection_glyph":{"id":"4e5068dc-3162-49a4-8946-59a0efb85147","type":"Line"},"selection_glyph":null},"id":"55a119fb-e583-4897-a3d1-b53f526303e8","type":"GlyphRenderer"},{"attributes":{},"id":"b376b1e6-3f85-4e47-a7e2-25875a1ddaff","type":"ToolEvents"},{"attributes":{"formatter":{"id":"b6f52db6-285c-4304-8078-ce9528d84fa0","type":"CategoricalTickFormatter"},"major_label_orientation":1,"plot":{"id":"cf85504c-4c14-455c-a34f-eb6f13e68186","subtype":"Figure","type":"Plot"},"ticker":{"id":"cb6479f0-59fe-435b-af53-d15c6283f435","type":"CategoricalTicker"}},"id":"6525acf4-3079-4a40-98e6-488d0c252697","type":"CategoricalAxis"},{"attributes":{},"id":"b6f52db6-285c-4304-8078-ce9528d84fa0","type":"CategoricalTickFormatter"},{"attributes":{"bottom_units":"screen","fill_alpha":{"value":0.5},"fill_color":{"value":"lightgrey"},"left_units":"screen","level":"overlay","line_alpha":{"value":1.0},"line_color":{"value":"black"},"line_dash":[4,4],"line_width":{"value":2},"plot":null,"render_mode":"css","right_units":"screen","top_units":"screen"},"id":"52a316fb-2dd5-4e7e-9865-365b861a5692","type":"BoxAnnotation"},{"attributes":{"callback":null},"id":"d7a270b8-16b6-4801-8378-fc35bf9adcc7","type":"DataRange1d"},{"attributes":{"active_drag":"auto","active_scroll":"auto","active_tap":"auto","tools":[{"id":"b80cdef7-983d-4b1e-b0ce-9a523bdd58ca","type":"PanTool"},{"id":"8be8a7db-636b-4486-979d-39021a61e339","type":"WheelZoomTool"},{"id":"1e5f757a-4bb8-407c-8960-0f9ddf9f9550","type":"BoxZoomTool"},{"id":"330d3cc0-1199-4287-9806-6f2ade567297","type":"SaveTool"},{"id":"b596be80-8911-4d78-8d0e-9c421b66dbd8","type":"ResetTool"},{"id":"f791276f-e7ff-4a4b-83a1-7e58ff5d16e0","type":"HelpTool"}]},"id":"83ab5bd2-7870-4290-9c59-e8928d39c238","type":"Toolbar"},{"attributes":{"below":[{"id":"6525acf4-3079-4a40-98e6-488d0c252697","type":"CategoricalAxis"}],"left":[{"id":"b91f5783-f051-45ac-be7b-37036eb2d6de","type":"LinearAxis"}],"plot_height":250,"plot_width":1000,"renderers":[{"id":"6525acf4-3079-4a40-98e6-488d0c252697","type":"CategoricalAxis"},{"id":"57fef3ad-83f5-401c-ae5e-5d7d660144cc","type":"Grid"},{"id":"b91f5783-f051-45ac-be7b-37036eb2d6de","type":"LinearAxis"},{"id":"d64e6dbd-25a3-4acd-8ecf-26e1778372a2","type":"Grid"},{"id":"52a316fb-2dd5-4e7e-9865-365b861a5692","type":"BoxAnnotation"},{"id":"55a119fb-e583-4897-a3d1-b53f526303e8","type":"GlyphRenderer"}],"title":{"id":"1f093269-4e51-4447-9ca0-8fe24d432122","type":"Title"},"tool_events":{"id":"b376b1e6-3f85-4e47-a7e2-25875a1ddaff","type":"ToolEvents"},"toolbar":{"id":"83ab5bd2-7870-4290-9c59-e8928d39c238","type":"Toolbar"},"x_range":{"id":"c3be40a7-751e-4245-9caa-9fd651a03f90","type":"FactorRange"},"y_range":{"id":"d7a270b8-16b6-4801-8378-fc35bf9adcc7","type":"DataRange1d"}},"id":"cf85504c-4c14-455c-a34f-eb6f13e68186","subtype":"Figure","type":"Plot"},{"attributes":{"plot":null,"text":"Stacked Loan Amount by Fiscal Quarter"},"id":"1f093269-4e51-4447-9ca0-8fe24d432122","type":"Title"},{"attributes":{"formatter":{"id":"d9332db8-0bf8-48ee-9977-6545daf3edfe","type":"BasicTickFormatter"},"plot":{"id":"cf85504c-4c14-455c-a34f-eb6f13e68186","subtype":"Figure","type":"Plot"},"ticker":{"id":"7b58627c-1b8d-49dd-87fb-efaf89354f53","type":"BasicTicker"}},"id":"b91f5783-f051-45ac-be7b-37036eb2d6de","type":"LinearAxis"},{"attributes":{},"id":"d9332db8-0bf8-48ee-9977-6545daf3edfe","type":"BasicTickFormatter"},{"attributes":{"callback":null,"column_names":["x","y"],"data":{"x":["2016Q3","2017Q1","2018Q1","2019Q1","2020Q4","2021Q1","2021Q2","2021Q3","2021Q4","2022Q1","2022Q2","2022Q3","2022Q4","2023Q1","2023Q2","2023Q3","2023Q4","2024Q1","2024Q2","2024Q3","2025Q1","2025Q2","2025Q3","2026Q1","2026Q2","2026Q3","2026Q4","2027Q1","2027Q2","2027Q3","2027Q4","2028Q1","2028Q2","2028Q3","2028Q4","2029Q1","2029Q2","2029Q3","2030Q3","2031Q2","2031Q3","2031Q4","2032Q1","2032Q2","2034Q3","2035Q3","2036Q1","2036Q2","2036Q3","2036Q4","2037Q1","2037Q2","2037Q3","2037Q4","2042Q1","2046Q1","2046Q3","2046Q4","2047Q1"],"y":{"__ndarray__":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAH4Xr8X30QEEfhevRiEsuQTQzM4tZNmxBHoXroXiqUEGuR+HyzmpwQShcjxL9BV1Bw/UopD3jaUEzMzNbrZFzQetRuD5gjEdBw/UonE/1ZEHiehSGaUt9QTQzMxWG7ZJB4XoUx5kyq0E+CtcsEzWhQYXrUaDDXqdBrkfhUXVzmEE+CtfLjjBmQR6F63EK7V1BcD0K9xSKUUHiehQexFlvQT0K19a/sKdBheuRxR4nskFlZmask9y8QXkUrqrydr1BXY/CSorGvkGPwvXcZI62QQAAAAAAAAAAZmZmJp2SQUEK16PYoJF6QUjheojE5oBBSOF6eVvIoUEK16MwvPKJQc3MzB7ufYtBZmbmH6NVoUEAAAAAAAAAANejcP35+jtBcT0K1+LXakEK16PwdpotQczMzKAScI9BAAAASJQObUEpXI+GMf11QT0K1yO7sjVBFK5HsfExYEEUrkenuDWLQcL1qMayHKNBNDMzb3FpnkGF61Fv1SGbQUjhek1D5aFB//9/wCMQpkEAAAAAAAAAAPYoXI+3KShBUrgeBaBBL0EfhesRQoNDQUjhevROPUBBwvUorLmAa0EAAABwC2ZaQQ==","dtype":"float64","shape":[59]}}},"id":"aaa3e294-a518-4ed7-a235-446194141cff","type":"ColumnDataSource"},{"attributes":{"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"x":{"field":"x"},"y":{"field":"y"}},"id":"4e5068dc-3162-49a4-8946-59a0efb85147","type":"Line"},{"attributes":{"plot":{"id":"cf85504c-4c14-455c-a34f-eb6f13e68186","subtype":"Figure","type":"Plot"},"ticker":{"id":"cb6479f0-59fe-435b-af53-d15c6283f435","type":"CategoricalTicker"}},"id":"57fef3ad-83f5-401c-ae5e-5d7d660144cc","type":"Grid"},{"attributes":{"callback":null,"factors":["2016Q3","2017Q1","2018Q1","2019Q1","2020Q4","2021Q1","2021Q2","2021Q3","2021Q4","2022Q1","2022Q2","2022Q3","2022Q4","2023Q1","2023Q2","2023Q3","2023Q4","2024Q1","2024Q2","2024Q3","2025Q1","2025Q2","2025Q3","2026Q1","2026Q2","2026Q3","2026Q4","2027Q1","2027Q2","2027Q3","2027Q4","2028Q1","2028Q2","2028Q3","2028Q4","2029Q1","2029Q2","2029Q3","2030Q3","2031Q2","2031Q3","2031Q4","2032Q1","2032Q2","2034Q3","2035Q3","2036Q1","2036Q2","2036Q3","2036Q4","2037Q1","2037Q2","2037Q3","2037Q4","2042Q1","2046Q1","2046Q3","2046Q4","2047Q1"]},"id":"c3be40a7-751e-4245-9caa-9fd651a03f90","type":"FactorRange"}],"root_ids":["cf85504c-4c14-455c-a34f-eb6f13e68186"]},"title":"Bokeh Application","version":"0.12.4"}};
            var render_items = [{"docid":"32a9e915-6572-45c7-b10a-a6574c6a385e","elementid":"cd8723a0-6912-4708-9629-bc4e0b658471","modelid":"cf85504c-4c14-455c-a34f-eb6f13e68186"}];
            
            Bokeh.embed.embed_items(docs_json, render_items);
          };
          if (document.readyState != "loading") fn();
          else document.addEventListener("DOMContentLoaded", fn);
        })();
      },
      function(Bokeh) {
      }
    ];
  
    function run_inline_js() {
      
      if ((window.Bokeh !== undefined) || (force === true)) {
        for (var i = 0; i < inline_js.length; i++) {
          inline_js[i](window.Bokeh);
        }if (force === true) {
          display_loaded();
        }} else if (Date.now() < window._bokeh_timeout) {
        setTimeout(run_inline_js, 100);
      } else if (!window._bokeh_failed_load) {
        console.log("Bokeh: BokehJS failed to load within specified timeout.");
        window._bokeh_failed_load = true;
      } else if (force !== true) {
        var cell = $(document.getElementById("cd8723a0-6912-4708-9629-bc4e0b658471")).parents('.cell').data().cell;
        cell.output_area.append_execute_result(NB_LOAD_WARNING)
      }
  
    }
  
    if (window._bokeh_is_loading === 0) {
      console.log("Bokeh: BokehJS loaded, going straight to plotting");
      run_inline_js();
    } else {
      load_libs(js_urls, function() {
        console.log("Bokeh: BokehJS plotting callback run at", now());
        run_inline_js();
      });
    }
  }(this));
</script>



```python
# 4
dat_4_origin = dat.copy()
dat_4_origin = dat_4_origin[dat_4_origin['Property Value'] != 0]
dat_4_origin['ratio'] = dat_4_origin['Loan Amount']/dat_4_origin['Property Value']
dat_4_origin['Property Value Inverse'] = 1/dat_4_origin['Property Value']
dat_4_origin['Days to Maturity'] = (dat_4_origin['Maturity Date'].dt.date - datetime.today().date()).dt.days
dat_4 = dat_4_origin.drop(['Maturity Date',
                           'Loan Amount',
                           'Year Built',
                           'Zip','built year error signal','Fiscal Year', 'Fiscal Quarter', 'Fiscal Year Quarter'],axis = 1)


dat_4_dummies = pd.get_dummies(dat_4)

signal = np.random.uniform(size = len(dat_4_dummies))
dat_4_train = dat_4_dummies[signal<=0.7]
dat_4_test = dat_4_dummies[signal>0.7]

x_train = dat_4_train.drop('ratio',axis = 1)
x_test = dat_4_test.drop('ratio',axis = 1)
y_train = dat_4_train.ratio
y_test = dat_4_test.ratio

rf = RandomForestRegressor(n_estimators=500,n_jobs= 4)

param = {'max_depth':list(range(5,16))+[None]}
model = GridSearchCV(rf,param)
model.fit(x_train,y_train)
prediction_train = model.predict(x_train)
prediction_test = model.predict(x_test)
print("""
In Sample:
R2:{}
MSE:{}
Out of Sample:
R2:{}
MSE:{}
""".format(r2_score(y_train,prediction_train),
           mean_squared_error(y_train,prediction_train),
           r2_score(y_test,prediction_test),
           mean_squared_error(y_test,prediction_test)))
```




    GridSearchCV(cv=None, error_score='raise',
           estimator=RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='auto', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=1, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=4,
               oob_score=False, random_state=None, verbose=0, warm_start=False),
           fit_params=None, iid=True, n_jobs=1,
           param_grid={'max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, None]},
           pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
           scoring=None, verbose=0)



    
    In Sample:
    R2:0.46169847659372976
    MSE:0.03706599077045559
    Out of Sample:
    R2:0.2636431039750824
    MSE:0.05222322897562938
    

