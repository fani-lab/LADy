import React from "react";
import "@/styles/globals.css";
import { ChakraProvider } from "@chakra-ui/react";
import { AppProps } from "next/app";
export default function App({ Component, pageProps }: AppProps) {
  return (
    <ChakraProvider>
      <Component {...pageProps} />
    </ChakraProvider>
  );
}
