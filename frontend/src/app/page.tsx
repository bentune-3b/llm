'use client';

import { useState } from 'react';
import Image from 'next/image';

export default function ChatPage() {
  const [messages, setMessages] = useState<{ role: 'user' | 'bot'; text: string }[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: { role: 'user'; text: string } = { role: 'user', text: input };
    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);
    setInput('');
    setLoading(true);

    try {
      const response = await fetch('https://bentune-backend.onrender.com/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: input }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'API Error');
      }

      // Explicitly type the bot message
      const botMessage: { role: 'bot'; text: string } = {
        role: 'bot',
        text: data.answer ?? 'No response from model.',
      };

      setMessages([...updatedMessages, botMessage]);
    } catch (error: any) {
      setMessages([
        ...updatedMessages,
        { role: 'bot', text: 'Error: ' + error.message },
      ]);
    }

    setLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-black">
      {/* Header with logo */}
      <header className="flex items-center p-4 border-b bg-black shadow-md">
        <Image src="/logo.png" alt="Logo" width={275} height={275} className="mr-3" />
      </header>

      {/* Message Area */}
      <div className="flex-1 overflow-y-auto p-6">
        {messages.map((msg, i) => (
          <div key={i} className={`mb-4 flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div
              className={`max-w-md px-4 py-2 rounded-lg ${
                msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-300 text-gray-800'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        {loading && <div className="text-sm text-gray-500">Thinking...</div>}
      </div>

      {/* Input Box */}
      <div className="p-4 border-t bg-black">
        <textarea
          rows={1}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type your message..."
          className="w-full p-3 border rounded resize-none focus:outline-none focus:ring-2 focus:ring-blue-400"
        />
        <button
          onClick={sendMessage}
          className="mt-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded float-right"
          disabled={loading}
        >
          Send
        </button>
      </div>
    </div>
  );
}