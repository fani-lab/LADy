import "@/styles/globals.css";
import { ChakraProvider } from "@chakra-ui/react";
import { useEffect, useState } from "react";
import { Spinner } from "@chakra-ui/react";
export default function App({ Component, pageProps }) {
  return (
    <ChakraProvider>
      <Component {...pageProps} />
    </ChakraProvider>
  );
}
