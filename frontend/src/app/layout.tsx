import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "ChatBot UI",
  description: "A sleek and responsive chat interface",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased`}>
        <main className="min-h-screen flex items-center justify-center bg-[var(--background)] text-[var(--foreground)] transition-all duration-300">
          <div className="w-full max-w-3xl border rounded-xl shadow-xl overflow-hidden">
            {children}
          </div>
        </main>
      </body>
    </html>
  );
}
