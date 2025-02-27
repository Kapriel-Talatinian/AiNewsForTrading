using System;
using System.IO;
using System.Net.Http;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Json;
using System.Text;
using System.Threading.Tasks;
using cAlgo.API;

[Robot(AccessRights = AccessRights.FullAccess)]
public class NewsTradingBot : Robot
{
    private HttpClient _httpClient;

    [Parameter("API URL", DefaultValue = "http://localhost:5000")]
    public string ApiUrl { get; set; }

    [Parameter("Volume (Units)", DefaultValue = 10000)]
    public double Volume { get; set; }

    [Parameter("Confidence Threshold", DefaultValue = 0.7, MinValue = 0.1, MaxValue = 1.0)]
    public double ConfidenceThreshold { get; set; }

    protected override void OnStart()
    {
        _httpClient = new HttpClient 
        {
            Timeout = TimeSpan.FromSeconds(3)
        };
    }

    protected override async void OnBar()
    {
        try
        {
            var signal = await GetSignalAnalysis();
            
            if (IsValidSignal(signal))
            {
                ExecuteOrder(signal);
                Print($"Signal exécuté: {signal.Recommendation}");
            }
        }
        catch (Exception ex)
        {
            HandleError(ex);
        }
    }

    private async Task<TradingSignal> GetSignalAnalysis()
    {
        var requestData = new ApiRequest 
        {
            Datetime = Bars.OpenTimes.LastValue.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ"),
            Symbol = SymbolName
        };

        // Sérialisation avec DataContractJsonSerializer
        var serializer = new DataContractJsonSerializer(typeof(ApiRequest));
        using (var stream = new MemoryStream())
        {
            serializer.WriteObject(stream, requestData);
            var json = Encoding.UTF8.GetString(stream.ToArray());
            
            var response = await _httpClient.PostAsync(
                $"{ApiUrl}/analyze",
                new StringContent(json, Encoding.UTF8, "application/json")
            );

            // Désérialisation de la réponse
            var responseStream = await response.Content.ReadAsStreamAsync();
            var responseSerializer = new DataContractJsonSerializer(typeof(TradingSignal));
            return (TradingSignal)responseSerializer.ReadObject(responseStream);
        }
    }

    private bool IsValidSignal(TradingSignal signal)
    {
        return signal != null 
            && signal.Confidence >= ConfidenceThreshold
            && Array.Exists(new[] {"BUY", "SELL"}, x => x == signal.Recommendation.ToUpper());
    }

    private void ExecuteOrder(TradingSignal signal)
    {
        var tradeType = signal.Recommendation.ToUpper() == "BUY" ? TradeType.Buy : TradeType.Sell;
        ExecuteMarketOrder(tradeType, SymbolName, Volume);
    }

    private void HandleError(Exception ex)
    {
        Chart.DrawStaticText(
            "Error", 
            $"⚠️ {ex.Message.Split(':')[0]}", 
            VerticalAlignment.Top, 
            HorizontalAlignment.Right, 
            Color.Red
        );
    }

    [DataContract]
    public class TradingSignal
    {
        [DataMember(Name = "recommendation")]
        public string Recommendation { get; set; }

        [DataMember(Name = "confidence")]
        public double Confidence { get; set; }

        [DataMember(Name = "reason")]
        public string Reason { get; set; }

        [DataMember(Name = "timestamp")]
        public string Timestamp { get; set; }
    }

    [DataContract]
    public class ApiRequest
    {
        [DataMember(Name = "datetime")]
        public string Datetime { get; set; }

        [DataMember(Name = "symbol")]
        public string Symbol { get; set; }
    }

    protected override void OnStop()
    {
        _httpClient?.Dispose();
    }
}