library(shiny)
library(titanic)
library(caret)
library(class)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(scales)

# ── Data prep (runs once) ────────────────────────────────────────────────────
set.seed(18)
data(titanic_train)
titanic <- titanic_train[, c("Survived","Pclass","Sex","Age","Fare")]
titanic <- na.omit(titanic)
titanic$Survived <- factor(titanic$Survived)
titanic$Sex      <- factor(titanic$Sex)

idx   <- createDataPartition(titanic$Survived, p = 0.8, list = FALSE)
train <- titanic[ idx, ]
test  <- titanic[-idx, ]

train$SexNum <- ifelse(train$Sex == "female", 1, 0)
test$SexNum  <- ifelse(test$Sex  == "female", 1, 0)

pre    <- preProcess(train[, c("Age","Fare","Pclass","SexNum")], method = c("center","scale"))
trainS <- predict(pre, train)
testS  <- predict(pre, test)
feats  <- c("Age","Fare","Pclass","SexNum")

# Pre-compute kNN accuracy curve
k_range <- 1:60
acc_curve <- sapply(k_range, function(k) {
  p <- knn(trainS[, feats], testS[, feats], cl = trainS$Survived, k = k)
  mean(p == testS$Survived)
})

# CART trees
tree_full <- rpart(Survived ~ ., data = train, method = "class",
                   control = rpart.control(cp = 0, minsplit = 2))
cp_opt      <- tree_full$cptable[which.min(tree_full$cptable[, 4]), 1]
tree_pruned <- prune(tree_full, cp = cp_opt)
acc_full    <- round(mean(predict(tree_full,   test, type="class") == test$Survived), 4)
acc_pruned  <- round(mean(predict(tree_pruned, test, type="class") == test$Survived), 4)
surv_rate   <- round(mean(titanic$Survived == "1") * 100, 1)

# ── CSS ─────────────────────────────────────────────────────────────────────
css <- "
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #0d1117;
  --surface:  #161b22;
  --surface2: #1c2330;
  --border:   #30363d;
  --text:     #e6edf3;
  --muted:    #8b949e;
  --blue:     #58a6ff;
  --blue-dim: #1f3a5f;
  --gold:     #e3b341;
  --gold-dim: #3d2e0a;
  --teal:     #3fb950;
  --teal-dim: #0d2a13;
  --purple:   #bc8cff;
  --red:      #f85149;
  --r:        10px;
}

html, body { font-family: 'Space Grotesk', sans-serif; background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.6; min-height: 100vh; }

/* kill shiny chrome */
.container-fluid { padding: 0 !important; }

/* ── LAYOUT ── */
.dash-wrap    { display: flex; min-height: 100vh; }
.dash-sidebar { width: 230px; min-width: 230px; background: var(--surface); border-right: 1px solid var(--border); padding: 0; display: flex; flex-direction: column; position: fixed; height: 100vh; overflow-y: auto; z-index: 100; }
.dash-main    { margin-left: 230px; flex: 1; display: flex; flex-direction: column; min-height: 100vh; }

/* ── SIDEBAR ── */
.sb-logo { padding: 24px 20px 20px; border-bottom: 1px solid var(--border); }
.sb-icon { width: 36px; height: 36px; background: linear-gradient(135deg,var(--blue),var(--purple)); border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 18px; margin-bottom: 10px; }
.sb-title { font-size: 0.85rem; font-weight: 700; letter-spacing: 0.04em; color: var(--text); }
.sb-sub   { font-size: 0.7rem; color: var(--muted); margin-top: 2px; }
.sb-nav   { padding: 16px 12px; flex: 1; }
.sb-group-label { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); padding: 10px 8px 4px; }
.sb-item  { display: flex; align-items: center; gap: 9px; padding: 8px 10px; border-radius: 7px; font-size: 0.83rem; color: var(--muted); cursor: pointer; margin-bottom: 2px; transition: all 0.12s; border: none; background: none; width: 100%; text-align: left; }
.sb-item:hover  { background: var(--surface2); color: var(--text); }
.sb-item.active { background: var(--blue-dim); color: var(--blue); font-weight: 600; }
.sb-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; opacity: 0.5; flex-shrink: 0; }

/* ── TOPBAR ── */
.dash-topbar { background: var(--surface); border-bottom: 1px solid var(--border); padding: 14px 28px; display: flex; align-items: center; justify-content: space-between; position: sticky; top: 0; z-index: 50; }
.tb-title { font-size: 0.95rem; font-weight: 600; }
.tb-right { display: flex; gap: 8px; align-items: center; }
.pill { display: inline-flex; align-items: center; padding: 3px 11px; border-radius: 20px; font-size: 0.67rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; }
.pill-blue   { background: var(--blue-dim);  color: var(--blue); }
.pill-gold   { background: var(--gold-dim);  color: var(--gold); }
.pill-teal   { background: var(--teal-dim);  color: var(--teal); }
.pill-muted  { background: var(--surface2);  color: var(--muted); }

/* ── CONTENT ── */
.dash-body  { padding: 26px 28px 60px; flex: 1; }

/* ── STAT ROW ── */
.stat-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 26px; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: 18px 20px; position: relative; overflow: hidden; }
.stat-card::before { content:''; position: absolute; top:0; left:0; right:0; height: 3px; }
.sc-blue::before   { background: var(--blue); }
.sc-gold::before   { background: var(--gold); }
.sc-teal::before   { background: var(--teal); }
.sc-purple::before { background: var(--purple); }
.stat-lbl { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
.stat-val { font-size: 1.8rem; font-weight: 700; line-height: 1.1; margin-bottom: 3px; }
.sc-blue   .stat-val { color: var(--blue); }
.sc-gold   .stat-val { color: var(--gold); }
.sc-teal   .stat-val { color: var(--teal); }
.sc-purple .stat-val { color: var(--purple); }
.stat-sub { font-size: 0.7rem; color: var(--muted); }

/* ── SECTION HEADER ── */
.sec-hdr { display: flex; align-items: center; gap: 12px; margin: 32px 0 16px; padding-bottom: 12px; border-bottom: 1px solid var(--border); }
.sec-num { width: 28px; height: 28px; background: linear-gradient(135deg,var(--blue),var(--purple)); border-radius: 7px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; font-weight: 700; color: #fff; flex-shrink: 0; }
.sec-num.gold { background: linear-gradient(135deg,var(--gold),#d4820a); }
.sec-num.teal { background: linear-gradient(135deg,var(--teal),#1a7f2e); }
.sec-title { font-size: 1rem; font-weight: 600; }
.sec-badge { margin-left: auto; }

/* ── CARDS ── */
.card     { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: 20px 22px; margin-bottom: 14px; }
.card-ttl { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); margin-bottom: 14px; }
.two-col  { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; margin: 14px 0 24px; }

/* ── QA CARDS ── */
.qa-card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: 18px 20px; }
.qa-card.knn  { border-top: 2px solid var(--gold); }
.qa-card.cart { border-top: 2px solid var(--teal); }
.qa-q  { font-size: 0.63rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 9px; }
.qa-card.knn  .qa-q { color: var(--gold); }
.qa-card.cart .qa-q { color: var(--teal); }
.qa-body { font-size: 0.82rem; color: #c9d1d9; line-height: 1.7; }
.qa-body b { color: var(--text); font-weight: 600; }
.qa-body i { color: var(--blue); font-style: normal; }

/* ── SLIDER ── */
.shiny-input-container { margin-bottom: 0; }
.slider-wrap { padding: 10px 0 6px; }
.irs--shiny .irs-bar     { background: var(--blue) !important; border-color: var(--blue) !important; }
.irs--shiny .irs-handle  { background: var(--blue) !important; border-color: var(--blue) !important; }
.irs--shiny .irs-single  { background: var(--blue) !important; }
.irs--shiny .irs-line    { background: var(--border) !important; border-color: var(--border) !important; }
.irs--shiny .irs-min, .irs--shiny .irs-max, .irs--shiny .irs-from, .irs--shiny .irs-to { color: var(--muted) !important; background: var(--surface2) !important; }
label { color: var(--muted) !important; font-size: 0.72rem !important; font-weight: 600 !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }

/* ── CALLOUT ── */
.callout { background: linear-gradient(135deg,rgba(88,166,255,0.07),rgba(188,140,255,0.07)); border: 1px solid rgba(88,166,255,0.22); border-radius: var(--r); padding: 18px 22px; margin: 20px 0; display: flex; gap: 12px; }
.callout-icon { font-size: 1.1rem; flex-shrink: 0; margin-top: 2px; }
.callout-text { font-size: 0.83rem; color: #c9d1d9; line-height: 1.65; }
.callout-text b { color: var(--blue); }

/* ── VALUE BOX (shiny output) ── */
#knn_acc_box, #knn_k_box { color: var(--gold) !important; }

/* plots bg */
.shiny-plot-output { border-radius: 6px; overflow: hidden; }
"

# ── UI ───────────────────────────────────────────────────────────────────────
ui <- fluidPage(
  tags$head(
    tags$style(HTML(css)),
    tags$title("Titanic ML Dashboard")
  ),
  
  div(class = "dash-wrap",
      
      # ── Sidebar ──────────────────────────────────────────────────────────────
      div(class = "dash-sidebar",
          div(class = "sb-logo",
              div(class = "sb-icon", "🚢"),
              div(class = "sb-title", "Titanic ML"),
              div(class = "sb-sub",  "Group 8 · kNN & CART")
          ),
          div(class = "sb-nav",
              div(class = "sb-group-label", "Overview"),
              div(class = "sb-item active", tags$span(class="sb-dot"), "Dashboard")
          )
      ),
      
      # ── Main ─────────────────────────────────────────────────────────────────
      div(class = "dash-main",
          
          # Topbar
          div(class = "dash-topbar",
              div(class = "tb-title", "Titanic Survival — kNN & CART Analysis"),
              div(class = "tb-right",
                  span(class="pill pill-blue", "set.seed(18)"),
                  span(class="pill pill-muted", as.character(Sys.Date()))
              )
          ),
          
          div(class = "dash-body",
              
              # ── Stat row ─────────────────────────────────────────────────────────
              div(class = "stat-row",
                  div(class = "stat-card sc-blue",
                      div(class="stat-lbl", "Training Rows"),
                      div(class="stat-val", nrow(train)),
                      div(class="stat-sub", "80 / 20 split")
                  ),
                  div(class = "stat-card sc-gold",
                      div(class="stat-lbl", "Best kNN Acc."),
                      div(class="stat-val", textOutput("best_acc_val", inline=TRUE)),
                      div(class="stat-sub", textOutput("best_k_val",   inline=TRUE))
                  ),
                  div(class = "stat-card sc-teal",
                      div(class="stat-lbl", "Pruned Tree Acc."),
                      div(class="stat-val", paste0(round(acc_pruned*100,1),"%")),
                      div(class="stat-sub", paste0("vs ", round(acc_full*100,1), "% unpruned"))
                  ),
                  div(class = "stat-card sc-purple",
                      div(class="stat-lbl", "Survival Rate"),
                      div(class="stat-val", paste0(surv_rate,"%")),
                      div(class="stat-sub", "full dataset")
                  )
              ),
              
              # ── Section 1: Data Cleaning & Scaling ───────────────────────────────
              div(class="sec-hdr",
                  div(class="sec-num", "1"),
                  div(class="sec-title", "Data Cleaning & Feature Scaling"),
                  span(class="sec-badge pill pill-blue", "Pre-processing")
              ),
              div(class="three-col",
                  div(class="card",
                      div(class="card-ttl", HTML("&#x1F9F9; Data Cleaning")),
                      div(style="font-size:0.83rem;color:#c9d1d9;line-height:1.7;",
                          HTML(paste0(
                            "Features selected: <b>Survived, Pclass, Sex, Age, Fare</b><br>",
                            "Rows with <b>NA values removed</b> via <code style='color:var(--blue);'>na.omit()</code><br>",
                            "<b>Survived</b> &rarr; factor (0/1)<br>",
                            "<b>Sex</b> &rarr; factor, then numeric (female=1, male=0)<br>",
                            "Final dataset: <b>", nrow(titanic), " rows</b>"
                          ))
                      )
                  ),
                  div(class="card",
                      div(class="card-ttl", HTML("&#x2696;&#xFE0F; Feature Scaling (kNN)")),
                      div(style="font-size:0.83rem;color:#c9d1d9;line-height:1.7;",
                          HTML(paste0(
                            "Method: <b>center + scale</b> (z-score standardisation)<br>",
                            "Applied to: <i>Age, Fare, Pclass, SexNum</i><br>",
                            "Fit on <b>training set only</b> via <code style='color:var(--blue);'>preProcess()</code><br>",
                            "Applied to test set via <code style='color:var(--blue);'>predict(pre, test)</code><br>",
                            "Prevents <b>high-magnitude features</b> (e.g. Fare) dominating distance"
                          ))
                      )
                  ),
                  div(class="card",
                      div(class="card-ttl", HTML("&#x2702;&#xFE0F; Train / Test Split")),
                      div(style="font-size:0.83rem;color:#c9d1d9;line-height:1.7;",
                          HTML(paste0(
                            "<b>80 / 20 stratified split</b> via <code style='color:var(--blue);'>createDataPartition()</code><br>",
                            "Stratified on <b>Survived</b> to preserve class balance<br>",
                            "Training rows: <b>", nrow(train), "</b> &nbsp;&middot;&nbsp; Test rows: <b>", nrow(test), "</b><br>",
                            "Seed: <code style='color:var(--blue);'>set.seed(18)</code> for reproducibility<br>",
                            "CART trained on <b>unscaled</b> data (tree-based, scale-invariant)"
                          ))
                      )
                  )
              ),
              
              # ── Section 2: kNN ───────────────────────────────────────────────────
              div(class="sec-hdr",
                  div(class="sec-num", "2"),
                  div(class="sec-title", "k-Nearest Neighbours"),
                  span(class="sec-badge pill pill-gold", "Interactive · live accuracy")
              ),
              
              div(class="two-col",
                  div(class="card",
                      div(class="card-ttl", "Select k Value"),
                      div(class="slider-wrap",
                          sliderInput("k_val", label="k (number of neighbours)",
                                      min=1, max=60, value=5, step=1, width="100%")
                      ),
                      br(),
                      div(class="card-ttl", "Live Result"),
                      div(style="display:flex;gap:20px;align-items:flex-end;margin-top:6px;",
                          div(
                            div(class="stat-lbl", "Test Accuracy"),
                            div(style="font-size:2rem;font-weight:700;color:var(--gold);",
                                textOutput("live_acc", inline=TRUE))
                          ),
                          div(
                            div(class="stat-lbl", "Correct / Total"),
                            div(style="font-size:1.1rem;color:var(--muted);margin-top:4px;",
                                textOutput("live_correct", inline=TRUE))
                          )
                      )
                  ),
                  div(class="card",
                      div(class="card-ttl", "Accuracy Curve (k = 1 to 60)"),
                      plotOutput("knn_curve", height="220px")
                  )
              ),
              
              # kNN Q&A
              div(class="sec-hdr", style="margin-top:28px;",
                  div(class="sec-num gold", "?"),
                  div(class="sec-title", "kNN — Conceptual Questions")
              ),
              div(class="three-col",
                  div(class="qa-card knn",
                      div(class="qa-q", "Q a · Accuracy drop at large k"),
                      div(class="qa-body", HTML("When <b>k is very large</b>, predictions average over many distant neighbours from different classes. The model ignores local structure and drifts toward predicting the <i>majority class</i> for every observation — classic <b>underfitting</b> (high bias) that lowers test accuracy."))
                  ),
                  div(class="qa-card knn",
                      div(class="qa-q", "Q b · Why scaling is required"),
                      div(class="qa-body", HTML("kNN relies on <b>Euclidean distance</b>. Without scaling, large-valued features (e.g. <i>Fare</i> up to £500+) dominate the distance calculation and drown out small-ranged ones (e.g. <i>Pclass</i> ∈ {1,2,3}). Standardising to zero mean &amp; unit variance gives every feature <b>equal influence</b>."))
                  ),
                  div(class="qa-card knn",
                      div(class="qa-q", "Q c · Small k vs large k"),
                      div(class="qa-body", HTML("<b>Small k</b> → flexible, jagged boundary → low bias, <i>high variance</i> → overfits noise.<br><br><b>Large k</b> → smooth boundary → low variance, <i>high bias</i> → underfits real patterns.<br><br>Optimal k is found via <b>cross-validation</b>."))
                  )
              ),
              
              # ── Section 2: CART ──────────────────────────────────────────────────
              div(class="sec-hdr",
                  div(class="sec-num", "3"),
                  div(class="sec-title", "CART — Decision Tree"),
                  span(class="sec-badge pill pill-teal", "Unpruned vs Pruned")
              ),
              
              div(class="two-col",
                  div(class="card",
                      div(class="card-ttl", "Tree Selection"),
                      radioButtons("tree_type", label="Display tree:",
                                   choices = c("Pruned (optimal CP)" = "pruned",
                                               "Unpruned (cp = 0)"   = "full"),
                                   selected = "pruned",
                                   inline = FALSE),
                      br(),
                      div(class="card-ttl", "Accuracy"),
                      uiOutput("cart_acc_table")
                  ),
                  div(class="card",
                      div(class="card-ttl", "CP Table (cross-validation error)"),
                      uiOutput("cp_table")
                  )
              ),
              
              div(class="card",
                  div(class="card-ttl", "📊 Decision Tree Visualisation"),
                  plotOutput("tree_plot", height="480px")
              ),
              
              # CART Q&A
              div(class="sec-hdr", style="margin-top:28px;",
                  div(class="sec-num teal", "?"),
                  div(class="sec-title", "CART — Conceptual Questions")
              ),
              div(class="three-col",
                  div(class="qa-card cart",
                      div(class="qa-q", "Q a · Easier to interpret"),
                      div(class="qa-body", HTML("The <b>pruned tree</b> is far easier to interpret. Pruning removes low-value branches, leaving 3–6 nodes readable root-to-leaf in seconds. The unpruned tree has dozens of splits — many reflecting noise — making it hard to audit or explain."))
                  ),
                  div(class="qa-card cart",
                      div(class="qa-q", "Q b · Better on test data"),
                      div(class="qa-body", HTML("The <b>pruned tree</b> generalises better. Cutting branches that memorised training noise reduces variance and transfers accuracy more reliably to unseen data. The comparison table above confirms this directly."))
                  ),
                  div(class="qa-card cart",
                      div(class="qa-q", "Q c · More likely to overfit"),
                      div(class="qa-body", HTML("The <b>unpruned tree</b> (cp = 0, minsplit = 2) overfits heavily — growing until each leaf is near-pure, capturing noise in training data. Near-perfect training accuracy that collapses on test data is the hallmark of <i>high variance</i>."))
                  )
              ),
              
              # Summary
              div(class="callout",
                  div(class="callout-icon", "💡"),
                  div(class="callout-text", HTML("<b>Summary:</b> kNN requires careful choice of k and feature scaling — without it, high-magnitude variables like Fare dominate distance calculations. CART is naturally interpretable but must be pruned to prevent overfitting. Both models agree that <b>Sex</b>, <b>Pclass</b>, and <b>Age</b> are the strongest predictors of Titanic survival."))
              )
              
          ) # /dash-body
      )   # /dash-main
  )     # /dash-wrap
)

# ── Server ───────────────────────────────────────────────────────────────────
server <- function(input, output, session) {
  
  # Reactive kNN prediction
  knn_pred <- reactive({
    knn(trainS[, feats], testS[, feats], cl = trainS$Survived, k = input$k_val)
  })
  
  output$live_acc <- renderText({
    acc <- mean(knn_pred() == testS$Survived)
    paste0(round(acc * 100, 1), "%")
  })
  
  output$live_correct <- renderText({
    correct <- sum(knn_pred() == testS$Survived)
    paste0(correct, " / ", nrow(testS))
  })
  
  output$best_acc_val <- renderText({
    paste0(round(max(acc_curve) * 100, 1), "%")
  })
  
  output$best_k_val <- renderText({
    paste0("at k = ", k_range[which.max(acc_curve)])
  })
  
  # kNN accuracy curve
  output$knn_curve <- renderPlot({
    current_k   <- input$k_val
    current_acc <- acc_curve[current_k]
    
    df <- data.frame(k = k_range, acc = acc_curve)
    ggplot(df, aes(k, acc)) +
      geom_line(color = "#58a6ff", linewidth = 1) +
      geom_point(data = data.frame(k=current_k, acc=current_acc),
                 aes(k, acc), color = "#e3b341", size = 4) +
      geom_vline(xintercept = current_k, color = "#e3b341",
                 linetype = "dashed", linewidth = 0.6, alpha = 0.6) +
      scale_y_continuous(labels = scales::percent_format(1),
                         limits = c(0.6, 0.88)) +
      scale_x_continuous(breaks = c(1, 10, 20, 30, 40, 50, 60)) +
      labs(x = "k", y = "Test Accuracy") +
      theme_minimal(base_family = "sans") +
      theme(
        plot.background  = element_rect(fill = "#161b22", color = NA),
        panel.background = element_rect(fill = "#161b22", color = NA),
        panel.grid.major = element_line(color = "#30363d", linewidth = 0.4),
        panel.grid.minor = element_blank(),
        axis.text  = element_text(color = "#8b949e", size = 10),
        axis.title = element_text(color = "#8b949e", size = 10),
        plot.margin = margin(8, 12, 8, 8)
      )
  }, bg = "#161b22")
  
  # CART accuracy table
  output$cart_acc_table <- renderUI({
    models  <- c("Unpruned", "Pruned")
    accs    <- c(acc_full, acc_pruned)
    pct     <- paste0(round(accs * 100, 1), "%")
    best    <- which.max(accs)
    
    header <- tags$tr(
      lapply(c("Model", "Accuracy", "Acc %"), function(h)
        tags$th(h, style="padding:6px 10px;text-align:left;color:#8b949e;
                          font-size:0.68rem;font-weight:700;letter-spacing:0.08em;
                          text-transform:uppercase;border-bottom:1px solid #30363d;")
      )
    )
    
    rows <- lapply(1:2, function(i) {
      is_best <- i == best
      row_bg  <- if (is_best) "background:#0d2a13;" else if (i %% 2 == 0) "background:#1c2330;" else ""
      val_col <- if (is_best) "color:#3fb950;font-weight:600;" else "color:#c9d1d9;"
      tags$tr(style = row_bg,
              tags$td(models[i], style=paste0("padding:5px 10px;font-size:0.8rem;", val_col)),
              tags$td(accs[i],   style=paste0("padding:5px 10px;font-size:0.8rem;font-family:monospace;", val_col)),
              tags$td(
                if (is_best) tagList(pct[i], tags$span(" ✓", style="color:#3fb950;margin-left:4px;"))
                else pct[i],
                style = paste0("padding:5px 10px;font-size:0.8rem;font-weight:700;", val_col)
              )
      )
    })
    
    tags$table(style="width:100%;border-collapse:collapse;",
               tags$thead(header),
               tags$tbody(rows)
    )
  })
  
  # CP table
  output$cp_table <- renderUI({
    # Use numeric indices — guaranteed regardless of rpart version or column naming
    ct <- tree_full$cptable   # numeric matrix: CP, nsplit, rel.error, xerror, xstd
    n  <- nrow(ct)
    cp_df <- data.frame(
      CP        = round(as.numeric(ct[seq_len(n), 1]), 5),
      nsplit    = as.integer(ct[seq_len(n), 2]),
      rel.error = round(as.numeric(ct[seq_len(n), 3]), 4),
      xerror    = round(as.numeric(ct[seq_len(n), 4]), 4),
      stringsAsFactors = FALSE
    )
    cp_df   <- head(cp_df[order(cp_df$xerror), ], 6)
    row.names(cp_df) <- NULL
    opt_idx <- which.min(cp_df$xerror)[1]
    
    header <- tags$tr(
      lapply(c("CP", "nsplit", "rel.error", "xerror"), function(h)
        tags$th(h, style = "padding:7px 12px;text-align:left;color:#8b949e;
                            font-size:0.67rem;font-weight:700;letter-spacing:0.09em;
                            text-transform:uppercase;border-bottom:2px solid #30363d;
                            background:#161b22;")
      )
    )
    
    rows <- lapply(seq_len(nrow(cp_df)), function(i) {
      is_opt  <- (i == opt_idx)
      row_bg  <- if (is_opt) "background:rgba(63,185,80,0.10);" else if (i %% 2 == 0) "background:#1c2330;" else "background:#161b22;"
      val_col <- if (is_opt) "color:#3fb950;font-weight:600;" else "color:#c9d1d9;"
      tags$tr(style = row_bg,
              tags$td(format(cp_df$CP[i], scientific = FALSE),
                      style = paste0("padding:6px 12px;font-size:0.79rem;font-family:monospace;", val_col)),
              tags$td(cp_df$nsplit[i],
                      style = paste0("padding:6px 12px;font-size:0.79rem;", val_col)),
              tags$td(cp_df$rel.error[i],
                      style = paste0("padding:6px 12px;font-size:0.79rem;", val_col)),
              tags$td(
                if (is_opt)
                  tagList(cp_df$xerror[i],
                          tags$span("  ✓ optimal",
                                    style = "font-size:0.67rem;color:#3fb950;margin-left:5px;font-weight:700;"))
                else cp_df$xerror[i],
                style = paste0("padding:6px 12px;font-size:0.79rem;", val_col)
              )
      )
    })
    
    tags$div(
      style = "border-radius:6px;overflow:hidden;border:1px solid #30363d;",
      tags$table(
        style = "width:100%;border-collapse:collapse;",
        tags$thead(header),
        tags$tbody(rows)
      )
    )
  })
  
  # Decision tree plot — light canvas for maximum clarity
  output$tree_plot <- renderPlot({
    tree <- if (input$tree_type == "pruned") tree_pruned else tree_full
    par(bg = "#ffffff", mar = c(0.5, 0.5, 1.5, 0.5))
    rpart.plot(
      tree,
      type          = 2,          # split label above node, class below
      extra         = 104,        # show % and n in each node
      fallen.leaves = TRUE,       # leaves at bottom for easy reading
      branch        = 0.4,        # angled branches
      branch.lwd    = 2,
      box.palette   = c("#d73027", "#4575b4"),  # red = died, blue = survived
      shadow.col    = "#cccccc",
      col           = "#111111",
      border.col    = "#555555",
      split.col     = "#333333",
      split.cex     = 1.05,
      nn.cex        = 0.85,
      tweak         = 1.2,
      compress      = TRUE,
      ycompress     = TRUE,
      main          = if (input$tree_type == "pruned") "Pruned Tree (optimal CP)" else "Unpruned Tree (cp = 0)",
      cex.main      = 1.1
    )
  }, bg = "#ffffff")
}

shinyApp(ui = ui, server = server)