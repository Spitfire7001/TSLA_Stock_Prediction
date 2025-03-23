# TSLA Stock Prediction Bot
The purpose of this Bot is to predict the next days closing stock price for the ticker TSLA. Using this information, the user can make a decision to BUY, SELL or HOLD their stocks.
## Installation Procedure

> [!NOTE]
> Setup will slightly vary if you are using Linux or Windows. This will be made clear.

To start, make a new directory where the repo can be cloned in to

```
mkdir yourDirName
cd yourDirName
```

Next, clone the repo into the directory that was just created
```
git clone https://github.com/Spitfire7001/TSLA_Stock_Prediction.git
```

Now a Python VENV will be created so the dependencies will not be installed system wide
```
python -m venv .venv
```
> [!IMPORTANT]
> This is where Windows and Linux Setup Slightly Varies

On Linux use:
```
source .venv/bin/activate
```
On Windows in Command Prompt:
```
.venv\Scripts\activate
```
Or on Windows in PowerShell
```
.\.venv\Scripts\Activate.ps1
```

Once the Virtual Environment is created and activated, the dependencies can be installed
```
pip install pandas numpy scikit-learn tensorflow yfinance matplotlib
```

Once all these steps are completed setup is done.

## Running Procedure
The required data will automatically be grabbed from the web when the bot is ran. An active internet connection is required for this to succeed.

Navigate to the repo directory:
```
cd TSLA_Stock_Prediction
```

To run the bot, use the following command:
```
python TSLAPredict.py
```