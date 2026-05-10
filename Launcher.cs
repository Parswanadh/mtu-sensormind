using System;
using System.Diagnostics;
using System.IO;
using System.Threading;

namespace MTUFleetLauncher
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.Title = "MTU SensorMind - Fleet Manager";
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine(@"
    __  __ _____ _    _   ____                           __  __ _           _ 
   |  \/  |_   _| |  | | / ___|  ___ _ __  ___  ___  _ _|  \/  (_)_ __   __| |
   | |\/| | | | | |  | | \___ \ / _ \ '_ \/ __|/ _ \| '__| |\/| | | '_ \ / _` |
   | |  | | | | | |__| |  ___) |  __/ | | \__ \ (_) | |  | |  | | | | | | (_| |
   |_|  |_| |_|  \____/  |____/ \___|_| |_|___/\___/|_|  |_|  |_|_|_| |_|\__,_|
                                                                               
   =============================================================================
             Rolls-Royce Power Systems | Equipment Health Co-Pilot
   =============================================================================
            ");
            Console.ResetColor();

            Console.WriteLine("[INFO] Initializing C# Fleet Manager Bridge...");
            Thread.Sleep(1000);

            string pythonPath = "python";
            string argsStr = "-m streamlit run dashboard/app.py --server.headless true";

            Console.WriteLine("[INFO] Launching Python Data Science Backend...");
            
            try
            {
                ProcessStartInfo startInfo = new ProcessStartInfo
                {
                    FileName = pythonPath,
                    Arguments = argsStr,
                    UseShellExecute = true,
                    CreateNoWindow = false
                };

                Process.Start(startInfo);
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine("\n[SUCCESS] MTU SensorMind Dashboard is launching.");
                Console.WriteLine("[SUCCESS] Opening http://localhost:8501 in your browser...");
                Console.ResetColor();
                
                Thread.Sleep(3000);
                Process.Start(new ProcessStartInfo("http://localhost:8501") { UseShellExecute = true });
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine("\n[ERROR] Failed to launch backend process.");
                Console.WriteLine(string.Format("[DETAILS] {0}", ex.Message));
                Console.ResetColor();
            }

            Console.WriteLine("\nPress any key to exit the C# Fleet Manager Bridge...");
            Console.ReadKey();
        }
    }
}
